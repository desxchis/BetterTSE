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

from test_scripts.evaluate_tedit_strength_effect import (
    _build_instruction_text,
    _compute_region_metrics,
    _load_eval_records,
    _resolve_runtime_config_path,
    _spearman_rho,
)
from tool.tedit_wrapper import TEditWrapper


SWAP_0_2 = {0: 2, 1: 1, 2: 0}


def _swap_label(label: int | None, mode: str) -> int | None:
    if label is None:
        return None
    value = int(label)
    if mode == "identity":
        return value
    if mode == "swap_weak_strong":
        return int(SWAP_0_2.get(value, value))
    raise ValueError(f"Unsupported swap mode: {mode}")


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
    label_mode: str,
    fixed_scalar: float | None,
    seed: int,
) -> dict[str, Any]:
    runtime_config_path = _resolve_runtime_config_path(model_path, config_path)
    records, attr_list = _load_eval_records(dataset_folder, split, max_samples)
    wrapper = TEditWrapper(model_path=model_path, config_path=runtime_config_path, device=device)

    per_sample: list[dict[str, Any]] = []
    per_label_rows: dict[str, list[dict[str, Any]]] = {"0": [], "1": [], "2": []}

    for eval_idx, record in enumerate(records):
        base = np.asarray(record["base"], dtype=np.float32)
        src_attrs = np.asarray(record["src_attrs"], dtype=np.int64)
        tgt_attrs = np.asarray(record["tgt_attrs"], dtype=np.int64)
        edit_mask = np.asarray(record["edit_mask"], dtype=bool)
        controls = sorted(record["controls"], key=lambda row: float(row["strength_scalar"]))

        label_axis: list[float] = []
        gains: list[float] = []
        sample_rows: list[dict[str, Any]] = []

        for control_idx, control in enumerate(controls):
            orig_label = None if control.get("strength_label") is None else int(control["strength_label"])
            runtime_label = _swap_label(orig_label, label_mode)
            runtime_scalar = float(control["strength_scalar"] if fixed_scalar is None else fixed_scalar)
            instruction_text = None
            target = np.asarray(control["target"], dtype=np.float32)

            torch.manual_seed(seed + eval_idx * 31 + control_idx)
            np.random.seed(seed + eval_idx * 31 + control_idx)
            edited_batch = wrapper.edit_time_series(
                ts=base,
                src_attrs=src_attrs,
                tgt_attrs=tgt_attrs,
                n_samples=1,
                sampler=sampler,
                edit_steps=edit_steps,
                strength_label=runtime_label,
                strength_scalar=runtime_scalar,
                instruction_text=instruction_text,
                return_diagnostics=False,
            )
            edited = np.asarray(edited_batch, dtype=np.float32).reshape(-1)
            metrics = _compute_region_metrics(base, target, edited, edit_mask)
            row = {
                "sample_idx": int(record["sample_idx"]),
                "original_strength_label": orig_label,
                "runtime_strength_label": runtime_label,
                "original_strength_scalar": float(control["strength_scalar"]),
                "runtime_strength_scalar": runtime_scalar,
                "strength_text": str(control.get("strength_text", "")),
                "original_instruction_text": control.get("instruction_text") or (
                    None if orig_label is None else _build_instruction_text(src_attrs, tgt_attrs, attr_list, orig_label)
                ),
                "edit_gain": metrics["edit_gain"],
                "target_mae_edit_region": metrics["target_mae_edit_region"],
                "bg_mae": metrics["bg_mae"],
            }
            sample_rows.append(row)
            if orig_label is not None:
                per_label_rows[str(orig_label)].append(row)
                if metrics["edit_gain"] is not None:
                    label_axis.append(float(orig_label))
                    gains.append(float(metrics["edit_gain"]))

        weak = next((row["edit_gain"] for row in sample_rows if row["original_strength_label"] == 0), None)
        strong = next((row["edit_gain"] for row in sample_rows if row["original_strength_label"] == 2), None)
        monotonic = bool(
            len(sample_rows) >= 3
            and all(
                sample_rows[idx]["edit_gain"] is not None
                and sample_rows[idx + 1]["edit_gain"] is not None
                and sample_rows[idx]["edit_gain"] <= sample_rows[idx + 1]["edit_gain"]
                for idx in range(len(sample_rows) - 1)
            )
        )
        per_sample.append(
            {
                "sample_idx": int(record["sample_idx"]),
                "label_mode": label_mode,
                "fixed_scalar": fixed_scalar,
                "rows": sample_rows,
                "strong_minus_weak_edit_gain": None if weak is None or strong is None else float(strong - weak),
                "spearman_rho_label_vs_edit_gain": _spearman_rho(label_axis, gains),
                "monotonic_pass": monotonic,
            }
        )

    edit_gain_mean = {
        label_key: float(np.mean([float(row["edit_gain"]) for row in rows if row["edit_gain"] is not None])) if rows else None
        for label_key, rows in per_label_rows.items()
    }
    weak = edit_gain_mean.get("0")
    strong = edit_gain_mean.get("2")
    monotonic_hits = [float(bool(row["monotonic_pass"])) for row in per_sample]
    rho_values = [row["spearman_rho_label_vs_edit_gain"] for row in per_sample if row["spearman_rho_label_vs_edit_gain"] is not None]

    return {
        "status": {"ok": True, "stage": "probe_tedit_strength_label_swap"},
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
            "label_mode": label_mode,
            "fixed_scalar": fixed_scalar,
            "seed": seed,
        },
        "summary": {
            "num_samples": len(per_sample),
            "edit_gain_mean_by_original_label": edit_gain_mean,
            "strong_minus_weak_edit_gain_mean": None if weak is None or strong is None else float(strong - weak),
            "family_spearman_rho_label_gain_mean": None if not rho_values else float(np.mean(rho_values)),
            "monotonic_hit_rate": None if not monotonic_hits else float(np.mean(monotonic_hits)),
        },
        "per_sample": per_sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe weak↔strong label swaps for TEdit strength control.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--dataset-folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--edit-steps", type=int, default=10)
    parser.add_argument("--sampler", default="ddim")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--label-mode", choices=["identity", "swap_weak_strong"], default="swap_weak_strong")
    parser.add_argument("--fixed-scalar", type=float, default=None)
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
        label_mode=str(args.label_mode),
        fixed_scalar=None if args.fixed_scalar is None else float(args.fixed_scalar),
        seed=int(args.seed),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
