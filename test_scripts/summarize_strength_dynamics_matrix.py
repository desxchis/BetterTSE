from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def _safe_mean(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _extract_layer_metric(pairwise_gap: Dict[str, Any], metric_name: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for layer_idx, payload in sorted(pairwise_gap.items(), key=lambda kv: int(kv[0])):
        value = payload.get(metric_name)
        if value is not None:
            out[str(layer_idx)] = float(value)
    return out


def _ratio_dict(num: Dict[str, float], denom: Dict[str, float]) -> Dict[str, float | None]:
    out: Dict[str, float | None] = {}
    for key in sorted(set(num) | set(denom), key=lambda x: int(x)):
        n = num.get(key)
        d = denom.get(key)
        if n is None or d is None or abs(d) < 1.0e-12:
            out[key] = None
        else:
            out[key] = float(n / d)
    return out


def summarize_experiment(exp_dir: Path) -> Dict[str, Any]:
    diag_jsonl = exp_dir / "0" / "trend_injection" / "strength_diagnostics.jsonl"
    eval_json = exp_dir / "heldout_monotonic_eval.json"
    modulation_json = exp_dir / "modulation_diagnostics.json"
    if not diag_jsonl.exists() or not eval_json.exists() or not modulation_json.exists():
        raise FileNotFoundError(f"Missing diagnostics under {exp_dir}")

    diag_rows = _read_jsonl(diag_jsonl)
    eval_payload = _read_json(eval_json)
    modulation_payload = _read_json(modulation_json)

    finetune_rows = [row.get("finetune_stats", {}) for row in diag_rows]
    grad_rows = [row.get("component_grad_norms", {}) for row in diag_rows]

    denoising_loss_mean = _safe_mean(row.get("denoising_loss") for row in finetune_rows)
    monotonic_loss_mean = _safe_mean(row.get("monotonic_loss") for row in finetune_rows)
    edit_region_l1_mean = _safe_mean(row.get("edit_region_l1") for row in finetune_rows)
    background_l1_mean = _safe_mean(row.get("background_l1") for row in finetune_rows)
    monotonic_active_family_rate_mean = _safe_mean(row.get("monotonic_active_family_rate") for row in finetune_rows)
    monotonic_grad_norm_mean = _safe_mean(row.get("monotonic_loss") for row in grad_rows)
    denoising_grad_norm_mean = _safe_mean(row.get("denoising_loss") for row in grad_rows)
    total_strength_grad_norm_mean = _safe_mean(
        _safe_mean(module.get("grad_norm") for module in row.get("strength_modules", []))
        for row in diag_rows
    )

    pairwise_gap = modulation_payload["summary"]["pairwise_abs_gap"].get("weak_vs_strong", {})
    modulation_delta = _extract_layer_metric(pairwise_gap, "delta_gamma_norm")
    output_delta = _extract_layer_metric(pairwise_gap, "output_norm")
    skip_delta = _extract_layer_metric(pairwise_gap, "skip_norm")
    residual_delta = _extract_layer_metric(pairwise_gap, "residual_norm")
    output_vs_modulation_decay = _ratio_dict(output_delta, modulation_delta)

    eval_summary = eval_payload["summary"]
    return {
        "experiment": exp_dir.name,
        "train_metrics": {
            "denoising_loss_mean": denoising_loss_mean,
            "monotonic_loss_mean": monotonic_loss_mean,
            "edit_region_l1_mean": edit_region_l1_mean,
            "background_l1_mean": background_l1_mean,
            "monotonic_loss_over_denoising_loss": (
                None
                if denoising_loss_mean in (None, 0.0) or monotonic_loss_mean is None
                else float(monotonic_loss_mean / denoising_loss_mean)
            ),
            "monotonic_active_family_rate_mean": monotonic_active_family_rate_mean,
            "monotonic_grad_norm_mean": monotonic_grad_norm_mean,
            "denoising_grad_norm_mean": denoising_grad_norm_mean,
            "monotonic_grad_over_denoising_grad": (
                None
                if denoising_grad_norm_mean in (None, 0.0) or monotonic_grad_norm_mean is None
                else float(monotonic_grad_norm_mean / denoising_grad_norm_mean)
            ),
            "total_strength_grad_norm_mean": total_strength_grad_norm_mean,
        },
        "heldout_metrics": {
            "strong_minus_weak_edit_gain_mean": eval_summary.get("strong_minus_weak_edit_gain_mean"),
            "monotonic_hit_rate": eval_summary.get("monotonic_hit_rate"),
            "preservation_pass_rate": eval_summary.get("preservation_pass_rate"),
        },
        "modulation_metrics": {
            "weak_vs_strong_delta_gamma_norm_by_layer": modulation_delta,
            "weak_vs_strong_output_norm_by_layer": output_delta,
            "weak_vs_strong_skip_norm_by_layer": skip_delta,
            "weak_vs_strong_residual_norm_by_layer": residual_delta,
            "output_over_delta_gamma_decay_by_layer": output_vs_modulation_decay,
        },
    }


def write_markdown(summary_rows: list[Dict[str, Any]], output_path: Path) -> None:
    lines = [
        "# Strength Dynamics Diagnostic Matrix",
        "",
        "| Experiment | mono/denoise | mono active rate | mono grad/denoise grad | strong-weak gain | hit rate | preservation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        train = row["train_metrics"]
        heldout = row["heldout_metrics"]
        lines.append(
            "| {experiment} | {mld} | {mar} | {mgr} | {swg} | {hit} | {pres} |".format(
                experiment=row["experiment"],
                mld=train["monotonic_loss_over_denoising_loss"],
                mar=train["monotonic_active_family_rate_mean"],
                mgr=train["monotonic_grad_over_denoising_grad"],
                swg=heldout["strong_minus_weak_edit_gain_mean"],
                hit=heldout["monotonic_hit_rate"],
                pres=heldout["preservation_pass_rate"],
            )
        )
        lines.append("")
        lines.append(f"## {row['experiment']}")
        lines.append("")
        lines.append(f"- weak_vs_strong delta_gamma by layer: `{row['modulation_metrics']['weak_vs_strong_delta_gamma_norm_by_layer']}`")
        lines.append(f"- weak_vs_strong output_norm by layer: `{row['modulation_metrics']['weak_vs_strong_output_norm_by_layer']}`")
        lines.append(f"- output/delta_gamma decay by layer: `{row['modulation_metrics']['output_over_delta_gamma_decay_by_layer']}`")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize strength dynamics diagnostic matrix outputs.")
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    matrix_dir = Path(args.matrix_dir)
    summary_rows = []
    for exp_dir in sorted([p for p in matrix_dir.iterdir() if p.is_dir()]):
        if exp_dir.name.startswith("."):
            continue
        try:
            summary_rows.append(summarize_experiment(exp_dir))
        except FileNotFoundError:
            continue

    output_json = Path(args.output_json) if args.output_json else matrix_dir / "diagnostic_matrix_summary.json"
    output_md = Path(args.output_md) if args.output_md else matrix_dir / "diagnostic_matrix_summary.md"
    output_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary_rows, output_md)
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
