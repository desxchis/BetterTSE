from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


REQUIRED_EVAL_KEYS = {"config", "summary", "per_sample", "per_strength_rows", "raw_final_rows"}
REQUIRED_MONOTONIC_KEYS = {"config", "summary", "families"}


ROOT = Path(__file__).resolve().parent.parent
TEDIT_ROOT = ROOT / "TEdit-main"


DEFAULT_VARIANTS = [
    {"name": "baseline", "freeze_backbone": False, "instruction_text_dropout_prob": 0.0},
    {"name": "freeze_backbone", "freeze_backbone": True, "instruction_text_dropout_prob": 0.0},
    {"name": "text_dropout", "freeze_backbone": False, "instruction_text_dropout_prob": 0.5},
    {"name": "freeze_backbone_and_text_dropout", "freeze_backbone": True, "instruction_text_dropout_prob": 0.5},
]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _tail_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _run(cmd: List[str], cwd: Path, env: Dict[str, str]) -> None:
    print(f"\n[run] cwd={cwd}")
    print("[run] " + " ".join(cmd))
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed with return code {code}\ncmd: {cmd}\ncwd: {cwd}\nstdout_tail:\n{stdout}\nstderr_tail:\n{stderr}".format(
                code=completed.returncode,
                cmd=" ".join(cmd),
                cwd=cwd,
                stdout=_tail_text(completed.stdout or ""),
                stderr=_tail_text(completed.stderr or ""),
            )
        )


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _require_file(path: Path, *, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing {label}: {path}")
    if not path.is_file():
        raise RuntimeError(f"Expected file for {label}, got: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Empty {label}: {path}")


def _validate_eval_artifact(path: Path, *, condition_mode: str) -> Dict[str, Any]:
    _require_file(path, label=f"eval artifact ({condition_mode})")
    payload = _load_json(path)
    missing = sorted(REQUIRED_EVAL_KEYS - set(payload.keys()))
    if missing:
        raise RuntimeError(f"Invalid eval artifact {path}: missing keys {missing}")
    summary = payload.get("summary") or {}
    config = payload.get("config") or {}
    if summary.get("condition_mode") != condition_mode:
        raise RuntimeError(f"Eval artifact {path} has condition_mode={summary.get('condition_mode')} expected={condition_mode}")
    if int(summary.get("n_samples") or 0) <= 0:
        raise RuntimeError(f"Eval artifact {path} has no samples")
    if not isinstance(payload.get("per_sample"), list) or not payload["per_sample"]:
        raise RuntimeError(f"Eval artifact {path} has empty per_sample")
    if not isinstance(payload.get("raw_final_rows"), list) or not payload["raw_final_rows"]:
        raise RuntimeError(f"Eval artifact {path} has empty raw_final_rows")
    strength_rows = payload.get("per_strength_rows") or {}
    required_strength_keys = {"0.0000", "0.5000", "1.0000"}
    if not required_strength_keys.issubset(set(strength_rows.keys())):
        raise RuntimeError(f"Eval artifact {path} missing strength rows for {sorted(required_strength_keys - set(strength_rows.keys()))}")
    for key in required_strength_keys:
        rows = strength_rows.get(key)
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"Eval artifact {path} has empty per_strength_rows[{key}]")
    for required_key in ("model_path", "config_path", "dataset_folder"):
        if not config.get(required_key):
            raise RuntimeError(f"Eval artifact {path} missing config.{required_key}")
    for metric_key in ("monotonic_hit_rate", "raw_monotonic_hit_rate", "final_monotonic_hit_rate"):
        if summary.get(metric_key) is None:
            raise RuntimeError(f"Eval artifact {path} missing summary.{metric_key}")
    return payload


def _validate_monotonic_artifact(path: Path) -> Dict[str, Any]:
    _require_file(path, label="monotonic artifact")
    payload = _load_json(path)
    missing = sorted(REQUIRED_MONOTONIC_KEYS - set(payload.keys()))
    if missing:
        raise RuntimeError(f"Invalid monotonic artifact {path}: missing keys {missing}")
    summary = payload.get("summary") or {}
    config = payload.get("config") or {}
    families = payload.get("families")
    if int(summary.get("num_families") or 0) <= 0:
        raise RuntimeError(f"Monotonic artifact {path} has no families")
    if not isinstance(families, list) or not families:
        raise RuntimeError(f"Monotonic artifact {path} has empty families")
    for required_key in ("benchmark_path", "model_path", "config_path"):
        if not config.get(required_key):
            raise RuntimeError(f"Monotonic artifact {path} missing config.{required_key}")
    for metric_key in ("adjacent_monotonic_pass_rate", "off_anchor_monotonic_pass_rate", "preservation_pass_rate"):
        if summary.get(metric_key) is None:
            raise RuntimeError(f"Monotonic artifact {path} missing summary.{metric_key}")
    return payload


def _build_variant_artifact_paths(variant_root: Path) -> Dict[str, Path]:
    return {
        "eval_both": variant_root / "eval_both.json",
        "eval_label_only": variant_root / "eval_label_only.json",
        "eval_text_only": variant_root / "eval_text_only.json",
        "monotonic_eval": variant_root / "monotonic_eval.json",
    }


def _finalize_variant_artifacts(paths: Dict[str, Path]) -> Dict[str, Any]:
    eval_payloads = {
        "both": _validate_eval_artifact(paths["eval_both"], condition_mode="both"),
        "label_only": _validate_eval_artifact(paths["eval_label_only"], condition_mode="label_only"),
        "text_only": _validate_eval_artifact(paths["eval_text_only"], condition_mode="text_only"),
    }
    monotonic_payload = _validate_monotonic_artifact(paths["monotonic_eval"])
    return {
        "eval": eval_payloads,
        "monotonic": monotonic_payload,
    }


def _artifact_status(paths: Dict[str, Path]) -> Dict[str, Any]:
    status: Dict[str, Any] = {}
    for name, path in paths.items():
        status[name] = {
            "path": str(path),
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() and path.is_file() else 0,
        }
    return status


def _validate_input_paths(*, ckpt_path: Path, model_config_path: Path, dataset_folder: Path, benchmark_path: Path) -> None:
    _require_file(ckpt_path, label="checkpoint")
    _require_file(model_config_path, label="model config")
    if not dataset_folder.exists() or not dataset_folder.is_dir():
        raise RuntimeError(f"Missing dataset folder: {dataset_folder}")
    _require_file(benchmark_path, label="benchmark")


def _audit_required_artifacts(report: Dict[str, Any]) -> None:
    missing: List[str] = []
    invalid: List[str] = []
    for variant in report.get("variants", []):
        artifact_status = variant.get("artifact_status") or {}
        for name, info in artifact_status.items():
            if not info.get("exists"):
                missing.append(f"{variant.get('name')}::{name}::{info.get('path')}")
            elif int(info.get("size") or 0) <= 0:
                invalid.append(f"{variant.get('name')}::{name}::{info.get('path')}")
    if missing or invalid:
        parts: List[str] = []
        if missing:
            parts.append("missing=" + ", ".join(missing))
        if invalid:
            parts.append("invalid=" + ", ".join(invalid))
        raise RuntimeError("Required ablation artifacts failed audit: " + " | ".join(parts))


def _extract_status_block(payload: Dict[str, Any]) -> Dict[str, Any]:
    status = payload.get("status") or {}
    return {
        "ok": bool(status.get("ok", False)),
        "stage": status.get("stage"),
    }


def _build_stage_status(*, ckpt_path: Path, model_config_path: Path, diagnostics_path: Path, artifact_paths: Dict[str, Path], eval_payloads: Dict[str, Dict[str, Any]], monotonic_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "checkpoint": {"path": str(ckpt_path), "exists": ckpt_path.exists()},
        "model_config": {"path": str(model_config_path), "exists": model_config_path.exists()},
        "diagnostics": {"path": str(diagnostics_path), "exists": diagnostics_path.exists()},
        "artifacts": _artifact_status(artifact_paths),
        "eval_status": {mode: _extract_status_block(payload) for mode, payload in eval_payloads.items()},
        "monotonic_status": _extract_status_block(monotonic_payload),
    }


def _build_manifest(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "output_dir": report.get("output_dir"),
        "variants": [
            {
                "name": row.get("name"),
                "checkpoint_path": row.get("checkpoint_path"),
                "stage_status": row.get("stage_status"),
            }
            for row in report.get("variants", [])
        ],
    }


def _write_manifest(output_dir: Path, report: Dict[str, Any]) -> Path:
    manifest_path = output_dir / "ablation_manifest.json"
    manifest_path.write_text(json.dumps(_build_manifest(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _extract_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    return payload.get("summary") or {}


def _mean(values: Iterable[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def _median(values: Iterable[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(statistics.median(filtered))


def _safe_div(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or abs(right) <= 1.0e-12:
        return None
    return float(left / right)


def _summarize_gradients(diag_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not diag_rows:
        return {}

    recent_rows = diag_rows[-min(10, len(diag_rows)) :]
    strength_grad = []
    backbone_grad = []
    strength_update = []
    backbone_update = []
    amplitude = []
    diffusion = []
    label_gain_corr = []
    family_gain_rho = []

    for row in recent_rows:
        grad_norm_by_param = row.get("grad_norm_by_param") or {}
        update_ratio_by_param = row.get("param_update_ratio_by_param") or {}
        for name, value in grad_norm_by_param.items():
            if value is None:
                continue
            if "strength_projector" in name or "strength_modulation" in name or "strength_input_projection" in name:
                strength_grad.append(float(value))
            elif "base_modulation" in name or "diffusion_projection" in name or "attr_projection" in name:
                backbone_grad.append(float(value))
        for name, value in update_ratio_by_param.items():
            if value is None:
                continue
            if "strength_projector" in name or "strength_modulation" in name or "strength_input_projection" in name:
                strength_update.append(float(value))
            elif "base_modulation" in name or "diffusion_projection" in name or "attr_projection" in name:
                backbone_update.append(float(value))

        loss_breakdown = row.get("loss_breakdown") or {}
        amplitude.append(loss_breakdown.get("weighted_amplitude_control_loss"))
        diffusion.append(loss_breakdown.get("weighted_diffusion_realism_loss"))

        generator_scalars = (
            (((row.get("training_batch_diagnostics") or {}).get("condition_path") or {}).get("generator") or {}).get("scalars")
            or {}
        )
        label_gain_corr.append(generator_scalars.get("train_edit_gain_scalar_corr"))
        family_gain_rho.append(generator_scalars.get("train_family_gain_scalar_spearman_mean"))

    strength_grad_mean = _mean(strength_grad)
    backbone_grad_mean = _mean(backbone_grad)
    strength_update_mean = _mean(strength_update)
    backbone_update_mean = _mean(backbone_update)
    amplitude_mean = _mean(amplitude)
    diffusion_mean = _mean(diffusion)

    return {
        "strength_grad_norm_mean_recent": strength_grad_mean,
        "backbone_neighbor_grad_norm_mean_recent": backbone_grad_mean,
        "strength_to_backbone_grad_ratio_recent": _safe_div(strength_grad_mean, backbone_grad_mean),
        "strength_update_ratio_mean_recent": strength_update_mean,
        "backbone_neighbor_update_ratio_mean_recent": backbone_update_mean,
        "strength_to_backbone_update_ratio_recent": _safe_div(strength_update_mean, backbone_update_mean),
        "weighted_amplitude_control_loss_mean_recent": amplitude_mean,
        "weighted_diffusion_realism_loss_mean_recent": diffusion_mean,
        "amplitude_to_diffusion_loss_ratio_recent": _safe_div(amplitude_mean, diffusion_mean),
        "train_edit_gain_scalar_corr_mean_recent": _mean(label_gain_corr),
        "train_family_gain_scalar_spearman_mean_recent": _mean(family_gain_rho),
    }


def _extract_eval_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "family_spearman_rho_strength_gain_mean": summary.get("family_spearman_rho_strength_gain_mean"),
        "gain_range_mean": summary.get("gain_range_mean"),
        "gain_calibration_mae_mean": summary.get("gain_calibration_mae_mean"),
        "raw_strong_minus_weak_mean": summary.get("raw_strong_minus_weak_mean"),
        "final_strong_minus_weak_mean": summary.get("final_strong_minus_weak_mean"),
        "monotonic_hit_rate": summary.get("monotonic_hit_rate"),
        "raw_monotonic_hit_rate": summary.get("raw_monotonic_hit_rate"),
        "final_monotonic_hit_rate": summary.get("final_monotonic_hit_rate"),
        "projector_signal_present": summary.get("projector_signal_present"),
        "preservation_flattens_strength": summary.get("preservation_flattens_strength"),
        "modulation_or_preservation_priority": summary.get("modulation_or_preservation_priority"),
    }


def _extract_monotonic_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "adjacent_monotonic_pass_rate": summary.get("adjacent_monotonic_pass_rate"),
        "off_anchor_monotonic_pass_rate": summary.get("off_anchor_monotonic_pass_rate"),
        "gain_range_mean": summary.get("gain_range_mean"),
        "family_spearman_rho_strength_gain_mean": summary.get("family_spearman_rho_strength_gain_mean"),
        "preservation_pass_rate": summary.get("preservation_pass_rate"),
    }


def _build_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Phase-1 Strength Ablation",
        "",
        f"- output_dir: `{report['output_dir']}`",
        f"- dataset_folder: `{report['dataset_folder']}`",
        f"- epochs: `{report['epochs']}`",
        f"- eval_max_samples: `{report['eval_max_samples']}`",
        f"- monotonic_max_families: `{report['monotonic_max_families']}`",
        "",
        "## Variant Summary",
        "",
        "| variant | freeze | text_dropout | both_rho | both_gain_range | label_rho | text_rho | grad_ratio | amp/diff | monotonic_adj |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["variants"]:
        both = row["eval"].get("both", {})
        label_only = row["eval"].get("label_only", {})
        text_only = row["eval"].get("text_only", {})
        diag = row.get("diagnostics_summary", {})
        mono = row.get("monotonic_eval", {})
        lines.append(
            "| {name} | {freeze} | {dropout:.2f} | {both_rho} | {both_range} | {label_rho} | {text_rho} | {grad_ratio} | {amp_ratio} | {mono_adj} |".format(
                name=row["name"],
                freeze=row["freeze_backbone"],
                dropout=float(row["instruction_text_dropout_prob"]),
                both_rho=_fmt(both.get("family_spearman_rho_strength_gain_mean")),
                both_range=_fmt(both.get("gain_range_mean")),
                label_rho=_fmt(label_only.get("family_spearman_rho_strength_gain_mean")),
                text_rho=_fmt(text_only.get("family_spearman_rho_strength_gain_mean")),
                grad_ratio=_fmt(diag.get("strength_to_backbone_grad_ratio_recent")),
                amp_ratio=_fmt(diag.get("amplitude_to_diffusion_loss_ratio_recent")),
                mono_adj=_fmt(mono.get("adjacent_monotonic_pass_rate")),
            )
        )
    lines.extend(["", "## Notes", ""])
    for row in report["variants"]:
        both = row["eval"].get("both", {})
        label_only = row["eval"].get("label_only", {})
        text_only = row["eval"].get("text_only", {})
        diag = row.get("diagnostics_summary", {})
        lines.append(
            (
                f"- `{row['name']}`: both/final strong-weak={_fmt(both.get('final_strong_minus_weak_mean'))}, "
                f"label-only rho={_fmt(label_only.get('family_spearman_rho_strength_gain_mean'))}, "
                f"text-only rho={_fmt(text_only.get('family_spearman_rho_strength_gain_mean'))}, "
                f"strength/backbone grad ratio={_fmt(diag.get('strength_to_backbone_grad_ratio_recent'))}, "
                f"amp/diff loss ratio={_fmt(diag.get('amplitude_to_diffusion_loss_ratio_recent'))}."
            )
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal phase-1 TEdit strength ablation matrix.")
    parser.add_argument("--dataset-folder", default="tmp/trend_strength_family_train_v1")
    parser.add_argument("--output-dir", default="tmp/phase1_strength_ablation")
    parser.add_argument("--pretrained-dir", default="TEdit-main/save/synthetic/pretrain_multi_weaver")
    parser.add_argument("--finetune-config-path", default="TEdit-main/configs/synthetic/finetune_strength_trend_family.yaml")
    parser.add_argument("--evaluate-config-path", default="TEdit-main/configs/synthetic/evaluate.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-max-samples", type=int, default=24)
    parser.add_argument("--monotonic-max-families", type=int, default=12)
    parser.add_argument("--edit-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--diagnostics-interval", type=int, default=5)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "dataset_folder": str(ROOT / args.dataset_folder),
        "epochs": int(args.epochs),
        "eval_max_samples": int(args.eval_max_samples),
        "monotonic_max_families": int(args.monotonic_max_families),
        "variants": [],
    }

    dataset_folder = (ROOT / args.dataset_folder).resolve()
    benchmark_path = (dataset_folder / "test.json").resolve()

    for variant in DEFAULT_VARIANTS:
        variant_name = str(variant["name"])
        variant_root = output_dir / variant_name
        train_output_dir = variant_root / "train"
        train_output_dir.mkdir(parents=True, exist_ok=True)
        artifact_paths = _build_variant_artifact_paths(variant_root)
        ckpt_path = train_output_dir / "0" / "trend_injection" / "ckpts" / "model_best.pth"
        model_config_path = train_output_dir / "0" / "trend_injection" / "model_configs.yaml"
        diagnostics_path = train_output_dir / "0" / "trend_injection" / "strength_diagnostics.jsonl"

        if args.force_train or not ckpt_path.exists():
            _run(
                [
                    sys.executable,
                    "run_finetune.py",
                    "--pretrained_dir",
                    str((ROOT / args.pretrained_dir).resolve()),
                    "--finetune_config_path",
                    str((ROOT / args.finetune_config_path).resolve()),
                    "--evaluate_config_path",
                    str((ROOT / args.evaluate_config_path).resolve()),
                    "--data_type",
                    "discrete_strength_family",
                    "--data_folder",
                    str(dataset_folder),
                    "--save_folder",
                    str(train_output_dir.resolve()),
                    "--n_runs",
                    "1",
                    "--epochs",
                    str(int(args.epochs)),
                    "--include_self",
                    "0",
                    "--freeze-backbone-for-strength",
                    "1" if variant["freeze_backbone"] else "0",
                    "--instruction-text-dropout-prob",
                    str(float(variant["instruction_text_dropout_prob"])),
                    "--strength-diagnostics",
                    "1",
                    "--strength-diagnostics-interval",
                    str(int(args.diagnostics_interval)),
                ],
                cwd=TEDIT_ROOT,
                env=env,
            )

        _validate_input_paths(
            ckpt_path=ckpt_path,
            model_config_path=model_config_path,
            dataset_folder=dataset_folder,
            benchmark_path=benchmark_path,
        )

        eval_payloads: Dict[str, Dict[str, Any]] = {}
        for condition_mode in ("both", "label_only", "text_only"):
            eval_output_path = artifact_paths[f"eval_{condition_mode}"]
            if args.force_eval or not eval_output_path.exists():
                _run(
                    [
                        sys.executable,
                        "test_scripts/evaluate_tedit_strength_effect.py",
                        "--model-path",
                        str(ckpt_path.resolve()),
                        "--config-path",
                        str(model_config_path.resolve()),
                        "--dataset-folder",
                        str(dataset_folder),
                        "--split",
                        "test",
                        "--max-samples",
                        str(int(args.eval_max_samples)),
                        "--edit-steps",
                        str(int(args.edit_steps)),
                        "--device",
                        str(args.device),
                        "--condition-mode",
                        condition_mode,
                        "--output",
                        str(eval_output_path.resolve()),
                    ],
                    cwd=ROOT,
                    env=env,
                )
            eval_payloads[condition_mode] = _validate_eval_artifact(eval_output_path, condition_mode=condition_mode)

        monotonic_output_path = artifact_paths["monotonic_eval"]
        if args.force_eval or not monotonic_output_path.exists():
            _run(
                [
                    sys.executable,
                    "test_scripts/run_tedit_trend_monotonic_eval.py",
                    "--benchmark",
                    str(benchmark_path),
                    "--model-path",
                    str(ckpt_path.resolve()),
                    "--config-path",
                    str(model_config_path.resolve()),
                    "--output",
                    str(monotonic_output_path.resolve()),
                    "--max-families",
                    str(int(args.monotonic_max_families)),
                    "--edit-steps",
                    str(int(args.edit_steps)),
                    "--device",
                    str(args.device),
                ],
                cwd=ROOT,
                env=env,
            )

        finalized = _finalize_variant_artifacts(artifact_paths)
        diagnostics_rows = _load_jsonl(diagnostics_path)
        row = {
            "name": variant_name,
            "freeze_backbone": bool(variant["freeze_backbone"]),
            "instruction_text_dropout_prob": float(variant["instruction_text_dropout_prob"]),
            "train_output_dir": str(train_output_dir),
            "checkpoint_path": str(ckpt_path),
            "diagnostics_path": str(diagnostics_path),
            "diagnostics_rows": int(len(diagnostics_rows)),
            "diagnostics_summary": _summarize_gradients(diagnostics_rows),
            "artifact_status": _artifact_status(artifact_paths),
            "stage_status": _build_stage_status(
                ckpt_path=ckpt_path,
                model_config_path=model_config_path,
                diagnostics_path=diagnostics_path,
                artifact_paths=artifact_paths,
                eval_payloads=finalized["eval"],
                monotonic_payload=finalized["monotonic"],
            ),
            "eval": {
                mode: _extract_eval_summary(_extract_summary(payload))
                for mode, payload in finalized["eval"].items()
            },
            "monotonic_eval": _extract_monotonic_summary(_extract_summary(finalized["monotonic"])),
        }
        report["variants"].append(row)

    _audit_required_artifacts(report)
    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path = output_dir / "summary.md"
    markdown_path.write_text(_build_markdown(report) + "\n", encoding="utf-8")
    manifest_path = _write_manifest(output_dir, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved: {report_path}")
    print(f"Saved: {markdown_path}")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
