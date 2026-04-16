from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_TEDIT_ROOT = _ROOT / "TEdit-main"
if str(_TEDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEDIT_ROOT))

from test_scripts.evaluate_tedit_strength_effect import _build_instruction_text, _compute_region_metrics, _load_eval_records, _resolve_runtime_config_path, _spearman_rho
from tool.tedit_wrapper import TEditWrapper

PROBE_STRENGTHS = [0.0, 0.5, 1.0]


def _condition_instruction(condition_mode: str, instruction_text: str | None) -> str | None:
    if condition_mode == "label_only":
        return None
    return instruction_text


def _condition_label(condition_mode: str, strength_label: int | None) -> int | None:
    if condition_mode == "text_only":
        return None
    return strength_label


def _effective_scalar(original_scalar: float, reverse_scalar: bool) -> float:
    if not reverse_scalar:
        return float(original_scalar)
    return float(1.0 - float(original_scalar))


def run_probe(
    *,
    model_path: str,
    config_path: str,
    dataset_folder: str,
    split: str,
    max_samples: int,
    edit_steps: int,
    sampler: str,
    device: str,
    condition_mode: str,
    reverse_scalar: bool,
    seed: int,
) -> dict[str, Any]:
    runtime_config_path = _resolve_runtime_config_path(model_path, config_path)
    records, attr_list = _load_eval_records(dataset_folder, split, max_samples)
    wrapper = TEditWrapper(model_path=model_path, config_path=runtime_config_path, device=device)

    per_sample: list[dict[str, Any]] = []
    per_strength_rows: dict[str, list[dict[str, Any]]] = {f"{value:.4f}": [] for value in PROBE_STRENGTHS}

    for eval_idx, record in enumerate(records):
        base = np.asarray(record["base"], dtype=np.float32)
        src_attrs = np.asarray(record["src_attrs"], dtype=np.int64)
        tgt_attrs = np.asarray(record["tgt_attrs"], dtype=np.int64)
        edit_mask = np.asarray(record["edit_mask"], dtype=bool)
        controls = sorted(record["controls"], key=lambda row: float(row["strength_scalar"]))

        gains: list[float] = []
        scalars: list[float] = []
        sweep_rows: list[dict[str, Any]] = []

        for strength_idx, control in enumerate(controls):
            original_scalar = float(control["strength_scalar"])
            effective_scalar = _effective_scalar(original_scalar, reverse_scalar=reverse_scalar)
            strength_label = _condition_label(condition_mode, control.get("strength_label"))
            instruction_text = control.get("instruction_text")
            if instruction_text is None and control.get("strength_label") is not None:
                instruction_text = _build_instruction_text(src_attrs, tgt_attrs, attr_list, int(control["strength_label"]))
            instruction_text = _condition_instruction(condition_mode, instruction_text)
            target = np.asarray(control["target"], dtype=np.float32)

            torch.manual_seed(seed + eval_idx * 31 + strength_idx)
            np.random.seed(seed + eval_idx * 31 + strength_idx)
            edited_batch, diagnostics = wrapper.edit_time_series(
                ts=base,
                src_attrs=src_attrs,
                tgt_attrs=tgt_attrs,
                n_samples=1,
                sampler=sampler,
                edit_steps=edit_steps,
                strength_label=strength_label,
                strength_scalar=effective_scalar,
                instruction_text=instruction_text,
                return_diagnostics=True,
            )
            edited = np.asarray(edited_batch, dtype=np.float32).reshape(-1)
            metrics = _compute_region_metrics(base, target, edited, edit_mask)
            row = {
                "sample_idx": int(record["sample_idx"]),
                "original_strength_scalar": original_scalar,
                "effective_strength_scalar": effective_scalar,
                "strength_label": strength_label,
                "instruction_text": instruction_text,
                "edit_gain": metrics["edit_gain"],
                "target_mae_edit_region": metrics["target_mae_edit_region"],
                "bg_mae": metrics["bg_mae"],
                "projector_signal_present": bool(
                    diagnostics.get("projector")
                    and any(
                        record.get("projector_output_pairwise_l2_by_scalar") or record.get("projector_output_pairwise_l2")
                        for record in diagnostics.get("projector", [])
                    )
                ),
            }
            sweep_rows.append(row)
            per_strength_rows[f"{original_scalar:.4f}"].append(row)
            if metrics["edit_gain"] is not None:
                scalars.append(original_scalar)
                gains.append(float(metrics["edit_gain"]))

        per_sample.append(
            {
                "sample_idx": int(record["sample_idx"]),
                "reverse_scalar": bool(reverse_scalar),
                "condition_mode": condition_mode,
                "rows": sweep_rows,
                "spearman_rho_original_strength_vs_edit_gain": _spearman_rho(scalars, gains),
                "monotonic_pass": bool(
                    len(gains) >= 2 and all(gains[idx + 1] >= gains[idx] for idx in range(len(gains) - 1))
                ),
            }
        )

    edit_gain_mean = {
        strength_key: float(np.mean([float(row["edit_gain"]) for row in rows if row["edit_gain"] is not None])) if rows else None
        for strength_key, rows in per_strength_rows.items()
    }
    bg_mae_mean = {
        strength_key: float(np.mean([float(row["bg_mae"]) for row in rows if row["bg_mae"] is not None])) if rows else None
        for strength_key, rows in per_strength_rows.items()
    }
    projector_signal_rate = {
        strength_key: float(np.mean([float(bool(row["projector_signal_present"])) for row in rows])) if rows else None
        for strength_key, rows in per_strength_rows.items()
    }

    weak = edit_gain_mean.get("0.0000")
    strong = edit_gain_mean.get("1.0000")
    monotonic_hits = [float(bool(row["monotonic_pass"])) for row in per_sample]
    rho_values = [row["spearman_rho_original_strength_vs_edit_gain"] for row in per_sample if row["spearman_rho_original_strength_vs_edit_gain"] is not None]

    return {
        "status": {"ok": True, "stage": "probe_reversed_strength_scalar"},
        "config": {
            "model_path": model_path,
            "config_path": runtime_config_path,
            "requested_config_path": config_path,
            "dataset_folder": dataset_folder,
            "split": split,
            "max_samples": max_samples,
            "edit_steps": edit_steps,
            "sampler": sampler,
            "device": device,
            "condition_mode": condition_mode,
            "reverse_scalar": reverse_scalar,
            "seed": seed,
        },
        "summary": {
            "num_samples": len(per_sample),
            "edit_gain_mean": edit_gain_mean,
            "bg_mae_mean": bg_mae_mean,
            "projector_signal_rate": projector_signal_rate,
            "strong_minus_weak_edit_gain_mean": None if weak is None or strong is None else float(strong - weak),
            "family_spearman_rho_strength_gain_mean": None if not rho_values else float(np.mean(rho_values)),
            "monotonic_hit_rate": None if not monotonic_hits else float(np.mean(monotonic_hits)),
        },
        "per_sample": per_sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe reversed scalar semantics for TEdit strength control.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--dataset-folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--edit-steps", type=int, default=10)
    parser.add_argument("--sampler", default="ddim")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--condition-mode", choices=["both", "label_only", "text_only"], default="label_only")
    parser.add_argument("--reverse-scalar", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    payload = run_probe(
        model_path=args.model_path,
        config_path=args.config_path,
        dataset_folder=args.dataset_folder,
        split=args.split,
        max_samples=int(args.max_samples),
        edit_steps=int(args.edit_steps),
        sampler=str(args.sampler),
        device=str(args.device),
        condition_mode=str(args.condition_mode),
        reverse_scalar=bool(args.reverse_scalar),
        seed=int(args.seed),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
