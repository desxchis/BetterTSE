from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_TEDIT_ROOT = _ROOT / "TEdit-main"
if str(_TEDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEDIT_ROOT))

from tool.tedit_wrapper import TEditWrapper


DEFAULT_STRENGTH_CONTROLS = [
    {"strength_label": 0, "strength_scalar": 0.0, "strength_text": "weak"},
    {"strength_label": 1, "strength_scalar": 0.5, "strength_text": "medium"},
    {"strength_label": 2, "strength_scalar": 1.0, "strength_text": "strong"},
]


def _to_jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _aggregate_mean(values):
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def _aggregate_nested_numeric_dict(records, key_candidates):
    bucket = {}
    for record in records:
        nested = None
        for key in key_candidates:
            value = record.get(key)
            if isinstance(value, dict):
                nested = value
                break
        if not isinstance(nested, dict):
            continue
        for sub_key, sub_value in nested.items():
            if isinstance(sub_value, (int, float)):
                bucket.setdefault(str(sub_key), []).append(float(sub_value))
    return {
        key: float(np.mean(values))
        for key, values in bucket.items() if values
    }


def _aggregate_nested_metric_by_strength(records, key_candidates):
    bucket = {}
    for record in records:
        nested = None
        for key in key_candidates:
            value = record.get(key)
            if isinstance(value, dict):
                nested = value
                break
        if not isinstance(nested, dict):
            continue
        for strength_key, payload in nested.items():
            if not isinstance(payload, dict):
                continue
            for metric_name, metric_value in payload.items():
                if isinstance(metric_value, (int, float)):
                    bucket.setdefault(str(strength_key), {}).setdefault(str(metric_name), []).append(float(metric_value))
    return {
        strength_key: {
            metric_name: float(np.mean(values))
            for metric_name, values in metric_dict.items() if values
        }
        for strength_key, metric_dict in bucket.items()
    }


def _extract_scalar_metrics(diagnostics: dict[str, object]) -> dict[str, object]:
    projector = diagnostics.get("projector") or []
    modulation_records = list(diagnostics.get("modulation_base") or []) + list(diagnostics.get("modulation_weaver") or [])
    stage_records = [
        record for record in modulation_records
        if isinstance(record, dict) and isinstance(record.get("stage_name"), str)
    ]
    modulation_core_records = [
        record for record in modulation_records
        if isinstance(record, dict) and record.get("stage_name") is None
    ]

    stage_summary = {}
    for record in stage_records:
        stage_name = str(record.get("stage_name"))
        stage_summary[stage_name] = {
            "mean_by_scalar": record.get("stage_mean_by_scalar") if isinstance(record.get("stage_mean_by_scalar"), dict) else None,
            "pairwise_l2_by_scalar": record.get("stage_pairwise_l2_by_scalar") if isinstance(record.get("stage_pairwise_l2_by_scalar"), dict) else None,
            "stage_norm_mean": record.get("stage_norm_mean"),
            "stage_abs_mean": record.get("stage_abs_mean"),
            "stage_feature_std_mean": record.get("stage_feature_std_mean"),
        }

    modulation_by_scalar = {}
    for tensor_name in ["delta_gamma", "delta_beta", "gamma_orig", "beta_orig", "gamma_final", "beta_final", "strength_cond"]:
        mean_key = f"{tensor_name}_mean_by_scalar"
        pairwise_key = f"{tensor_name}_pairwise_l2_by_scalar"
        modulation_by_scalar[tensor_name] = {
            "mean_by_scalar": _aggregate_nested_metric_by_strength(modulation_core_records, [mean_key]),
            "pairwise_l2_by_scalar": _aggregate_nested_numeric_dict(modulation_core_records, [pairwise_key]),
            "norm_mean": _aggregate_mean([record.get(f"{tensor_name}_norm_mean") for record in modulation_core_records]),
            "abs_mean": _aggregate_mean([record.get(f"{tensor_name}_abs_mean") for record in modulation_core_records]),
        }

    return {
        "projector_pairwise_l2": _aggregate_nested_numeric_dict(
            projector,
            ["projector_output_pairwise_l2_by_scalar", "projector_output_pairwise_l2"],
        ),
        "projector_strength_mean_norms": {
            str(k): v.get("norm") if isinstance(v, dict) else None
            for k, v in _aggregate_nested_metric_by_strength(
                projector,
                ["projector_output_mean_norm_by_scalar", "projector_output_mean_norm_by_strength"],
            ).items()
        },
        "modulation_delta_gamma_pairwise_l2": _aggregate_nested_numeric_dict(
            modulation_core_records,
            ["delta_gamma_pairwise_l2_by_scalar", "delta_gamma_pairwise_l2"],
        ),
        "modulation_delta_beta_pairwise_l2": _aggregate_nested_numeric_dict(
            modulation_core_records,
            ["delta_beta_pairwise_l2_by_scalar", "delta_beta_pairwise_l2"],
        ),
        "modulation_delta_gamma_abs_mean": _aggregate_mean([record.get("delta_gamma_abs_mean") for record in modulation_core_records]),
        "modulation_delta_beta_abs_mean": _aggregate_mean([record.get("delta_beta_abs_mean") for record in modulation_core_records]),
        "modulation_delta_gamma_over_base_mean": _aggregate_mean([record.get("delta_gamma_over_base_mean") for record in modulation_core_records]),
        "modulation_delta_beta_over_base_mean": _aggregate_mean([record.get("delta_beta_over_base_mean") for record in modulation_core_records]),
        "modulation_by_scalar": modulation_by_scalar,
        "stage_by_scalar": stage_summary,
    }


def _linf_diff(left: dict[str, object], right: dict[str, object], key: str) -> float | None:
    left_value = left.get(key)
    right_value = right.get(key)
    if left_value is None or right_value is None:
        return None
    return float(abs(float(left_value) - float(right_value)))


def _extract_stage_scalar_metric(stage_summary: dict[str, object], stage_name: str, scalar_key: str, metric_key: str) -> float | None:
    stage_payload = stage_summary.get(stage_name)
    if not isinstance(stage_payload, dict):
        return None
    mean_by_scalar = stage_payload.get("mean_by_scalar")
    if not isinstance(mean_by_scalar, dict):
        return None
    scalar_payload = mean_by_scalar.get(scalar_key)
    if not isinstance(scalar_payload, dict):
        return None
    value = scalar_payload.get(metric_key)
    if value is None:
        return None
    return float(value)


def _build_stage_transition_summary(stage_summary: dict[str, object], scalar_keys: list[str]) -> dict[str, object]:
    focus_stages = [
        "post_modulation",
        "post_mid_projection",
        "post_side_add",
        "mid_gate_logits",
        "mid_filter_logits",
        "mid_gate_activation",
        "mid_filter_activation",
        "mid_gated_output",
        "post_output_projection",
        "residual_merge",
    ]
    transitions = {}
    if len(scalar_keys) < 2:
        return transitions
    weakest_key = scalar_keys[0]
    strongest_key = scalar_keys[-1]
    for stage_name in focus_stages:
        weak_abs = _extract_stage_scalar_metric(stage_summary, stage_name, weakest_key, "mean_abs")
        strong_abs = _extract_stage_scalar_metric(stage_summary, stage_name, strongest_key, "mean_abs")
        weak_norm = _extract_stage_scalar_metric(stage_summary, stage_name, weakest_key, "norm")
        strong_norm = _extract_stage_scalar_metric(stage_summary, stage_name, strongest_key, "norm")
        if weak_abs is None and strong_abs is None and weak_norm is None and strong_norm is None:
            continue
        transitions[stage_name] = {
            "weak_abs": weak_abs,
            "strong_abs": strong_abs,
            "strong_minus_weak_abs": None if weak_abs is None or strong_abs is None else float(strong_abs - weak_abs),
            "weak_norm": weak_norm,
            "strong_norm": strong_norm,
            "strong_minus_weak_norm": None if weak_norm is None or strong_norm is None else float(strong_norm - weak_norm),
        }
    return transitions


def _region_mean_abs_delta(base: np.ndarray, edited: np.ndarray, mask: np.ndarray) -> float | None:
    edit_mask = np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(edit_mask):
        return None
    base = np.asarray(base, dtype=np.float32).reshape(-1)
    edited = np.asarray(edited, dtype=np.float32).reshape(-1)
    return float(np.mean(np.abs(edited[edit_mask] - base[edit_mask])))


def _load_probe_sample(dataset_folder: Path, split: str, requested_idx: int) -> dict[str, object]:
    meta_path = dataset_folder / "meta.json"
    if meta_path.exists():
        from data.discrete_strength_family import DiscreteStrengthFamilyDataset

        dataset = DiscreteStrengthFamilyDataset(str(dataset_folder))
        split_ds = dataset.get_split(split, include_self=True)
        idx = int(requested_idx) if requested_idx >= 0 else 0
        family = split_ds[idx]
        sample = family["samples"][0]
        controls = []
        for row in sorted(family["samples"], key=lambda value: float(value["strength_scalar"])):
            controls.append(
                {
                    "strength_label": None if row.get("strength_label") is None else int(row["strength_label"]),
                    "strength_scalar": float(row["strength_scalar"]),
                    "instruction_text": str(row.get("instruction_text")) if row.get("instruction_text") is not None else None,
                    "target": np.asarray(row["tgt_x"], dtype=np.float32).squeeze(-1),
                    "strength_text": str(row.get("strength_text", "medium")),
                }
            )
        return {
            "probe_idx": idx,
            "base": np.asarray(sample["src_x"], dtype=np.float32).squeeze(-1),
            "src_attrs": np.asarray(sample["src_attrs"], dtype=np.int64),
            "tgt_attrs": np.asarray(sample["tgt_attrs"], dtype=np.int64),
            "instruction_text": str(sample.get("instruction_text")) if sample.get("instruction_text") is not None else None,
            "mask_gt": np.asarray(sample["mask_gt"], dtype=np.float32).squeeze(-1) > 0.5,
            "controls": controls,
        }

    ts = np.load(dataset_folder / f"{split}_ts.npy")
    attrs = np.load(dataset_folder / f"{split}_attrs_idx.npy")
    idx = int(requested_idx) if requested_idx >= 0 else 0
    if requested_idx < 0:
        for cand_idx in range(len(attrs)):
            if not np.array_equal(attrs[cand_idx, 0], attrs[cand_idx, 1]):
                idx = cand_idx
                break
    instruction_text = None
    try:
        from data.synthetic_finetune import SyntheticDataset

        dataset = SyntheticDataset(str(dataset_folder))
        sample = dataset.get_split(split, include_self=True)[idx]
        if sample.get("instruction_text") is not None:
            instruction_text = str(sample.get("instruction_text"))
    except Exception:
        instruction_text = None
    return {
        "probe_idx": idx,
        "base": ts[idx, 0].astype(np.float32),
        "src_attrs": attrs[idx, 0].astype(np.int64),
        "tgt_attrs": attrs[idx, 1].astype(np.int64),
        "instruction_text": instruction_text,
        "mask_gt": np.abs(ts[idx, 1].astype(np.float32) - ts[idx, 0].astype(np.float32)) > 0,
        "controls": [
            {
                **control,
                "instruction_text": instruction_text,
                "target": ts[idx, 1].astype(np.float32),
            }
            for control in DEFAULT_STRENGTH_CONTROLS
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe whether TEdit strength injection changes diffusion outputs.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--dataset-folder", required=True)
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--sample-idx", type=int, default=-1)
    parser.add_argument("--edit-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task-id", type=int, default=-1, help="use -1 to omit task_id and probe strength-only path")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use-dataset-instruction", type=int, default=1)
    parser.add_argument("--condition-mode", default="both", choices=["both", "label_only", "text_only"], help="ablate whether strength comes from numeric scalar/label, text, or both")
    parser.add_argument("--enable-strength-diagnostics", type=int, default=1)
    parser.add_argument("--flip-beta-sign", type=int, default=0, choices=[0, 1])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_folder)
    sample_payload = _load_probe_sample(dataset_root, args.split, args.sample_idx)
    idx = int(sample_payload["probe_idx"])

    base = np.asarray(sample_payload["base"], dtype=np.float32)
    src_attrs = np.asarray(sample_payload["src_attrs"], dtype=np.int64)
    tgt_attrs = np.asarray(sample_payload["tgt_attrs"], dtype=np.int64)
    edit_mask = np.asarray(sample_payload["mask_gt"], dtype=bool)
    controls = sorted(list(sample_payload.get("controls") or []), key=lambda row: float(row["strength_scalar"]))
    if not controls:
        raise ValueError("probe sample does not contain any strength controls")

    instruction_texts = []
    for control in controls:
        text = control.get("instruction_text")
        if not bool(args.use_dataset_instruction):
            text = sample_payload.get("instruction_text")
        instruction_texts.append(text)

    numeric_strength_scalars = [float(control["strength_scalar"]) for control in controls]
    numeric_strength_labels = [control.get("strength_label") for control in controls]
    weakest_scalar = numeric_strength_scalars[0]
    weakest_label = next((int(value) for value in numeric_strength_labels if value is not None), 0)

    if args.condition_mode == "label_only":
        run_instruction_texts = None
        run_strength_scalars = numeric_strength_scalars
        run_strength_labels = None if any(value is None for value in numeric_strength_labels) else [int(value) for value in numeric_strength_labels]
    elif args.condition_mode == "text_only":
        run_instruction_texts = instruction_texts
        run_strength_scalars = [weakest_scalar] * len(controls)
        run_strength_labels = [weakest_label] * len(controls)
    else:
        run_instruction_texts = instruction_texts
        run_strength_scalars = numeric_strength_scalars
        run_strength_labels = None if any(value is None for value in numeric_strength_labels) else [int(value) for value in numeric_strength_labels]

    wrapper = TEditWrapper(model_path=args.model_path, config_path=args.config_path, device=args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    edited_batch, diagnostics = wrapper.edit_time_series(
        ts=np.repeat(base.reshape(1, -1), len(controls), axis=0),
        src_attrs=src_attrs,
        tgt_attrs=tgt_attrs,
        n_samples=1,
        sampler="ddim",
        edit_steps=args.edit_steps,
        strength_label=run_strength_labels,
        strength_scalar=run_strength_scalars,
        task_id=None if args.task_id < 0 else [int(args.task_id)] * len(controls),
        instruction_text=run_instruction_texts,
        return_diagnostics=True,
        enable_strength_diagnostics=bool(args.enable_strength_diagnostics),
        flip_beta_sign_inference=bool(args.flip_beta_sign),
    )
    model_diag = diagnostics["model"][0] if diagnostics.get("model") else {}
    raw_batch = np.asarray(model_diag.get("raw_reverse_output"), dtype=np.float32).squeeze(1)

    rows = []
    outputs = []
    for control_idx, control in enumerate(controls):
        edited = np.asarray(edited_batch[control_idx], dtype=np.float32)
        raw_output = np.asarray(raw_batch[control_idx], dtype=np.float32)
        outputs.append(edited)
        rows.append(
            {
                "strength_label": None if control.get("strength_label") is None else int(control["strength_label"]),
                "runtime_strength_label": None if run_strength_labels is None else int(run_strength_labels[control_idx]),
                "strength_scalar": float(control["strength_scalar"]),
                "runtime_strength_scalar": float(run_strength_scalars[control_idx]),
                "peak_abs_delta": float(np.max(np.abs(edited - base))),
                "mean_abs_delta": float(np.mean(np.abs(edited - base))),
                "sum_delta": float(np.sum(edited - base)),
                "raw_edit_region_mean_abs_delta": _region_mean_abs_delta(base, raw_output, edit_mask),
                "final_edit_region_mean_abs_delta": _region_mean_abs_delta(base, edited, edit_mask),
                "blend_gap_edit_region_mean_abs": _region_mean_abs_delta(raw_output, edited, edit_mask),
            }
        )

    scalar_diagnostics = _extract_scalar_metrics(diagnostics)
    scalar_keys = [f"{float(row['strength_scalar']):.4f}" for row in rows]
    stage_transition_summary = _build_stage_transition_summary(
        scalar_diagnostics.get("stage_by_scalar") or {},
        scalar_keys,
    )
    summary = {
        "condition_mode": args.condition_mode,
        "flip_beta_sign": bool(args.flip_beta_sign),
        "raw_edit_region_mean_abs_delta": {
            scalar_keys[row_idx]: row["raw_edit_region_mean_abs_delta"]
            for row_idx, row in enumerate(rows)
        },
        "final_edit_region_mean_abs_delta": {
            scalar_keys[row_idx]: row["final_edit_region_mean_abs_delta"]
            for row_idx, row in enumerate(rows)
        },
        "blend_gap_edit_region_mean_abs": {
            scalar_keys[row_idx]: row["blend_gap_edit_region_mean_abs"]
            for row_idx, row in enumerate(rows)
        },
        "projector_pairwise_l2": scalar_diagnostics.get("projector_pairwise_l2"),
        "projector_strength_mean_norms": scalar_diagnostics.get("projector_strength_mean_norms"),
        "modulation_delta_gamma_pairwise_l2": scalar_diagnostics.get("modulation_delta_gamma_pairwise_l2"),
        "modulation_delta_beta_pairwise_l2": scalar_diagnostics.get("modulation_delta_beta_pairwise_l2"),
        "modulation_delta_gamma_abs_mean": scalar_diagnostics.get("modulation_delta_gamma_abs_mean"),
        "modulation_delta_beta_abs_mean": scalar_diagnostics.get("modulation_delta_beta_abs_mean"),
        "modulation_delta_gamma_over_base_mean": scalar_diagnostics.get("modulation_delta_gamma_over_base_mean"),
        "modulation_delta_beta_over_base_mean": scalar_diagnostics.get("modulation_delta_beta_over_base_mean"),
        "modulation_by_scalar": scalar_diagnostics.get("modulation_by_scalar"),
        "stage_by_scalar": scalar_diagnostics.get("stage_by_scalar"),
        "stage_transition_summary": stage_transition_summary,
        "final_edit_delta_linf": {
            "weak_medium": _linf_diff(rows[0], rows[1], "final_edit_region_mean_abs_delta"),
            "medium_strong": _linf_diff(rows[1], rows[2], "final_edit_region_mean_abs_delta"),
            "weak_strong": _linf_diff(rows[0], rows[2], "final_edit_region_mean_abs_delta"),
        },
        "raw_edit_monotonic": bool(
            rows[0]["raw_edit_region_mean_abs_delta"] is not None
            and rows[1]["raw_edit_region_mean_abs_delta"] is not None
            and rows[2]["raw_edit_region_mean_abs_delta"] is not None
            and rows[0]["raw_edit_region_mean_abs_delta"] < rows[1]["raw_edit_region_mean_abs_delta"] < rows[2]["raw_edit_region_mean_abs_delta"]
        ),
        "final_edit_monotonic": bool(
            rows[0]["final_edit_region_mean_abs_delta"] is not None
            and rows[1]["final_edit_region_mean_abs_delta"] is not None
            and rows[2]["final_edit_region_mean_abs_delta"] is not None
            and rows[0]["final_edit_region_mean_abs_delta"] < rows[1]["final_edit_region_mean_abs_delta"] < rows[2]["final_edit_region_mean_abs_delta"]
        ),
    }
    payload = {
        "dataset_folder": str(dataset_root),
        "split": args.split,
        "probe_idx": int(idx),
        "src_attrs": src_attrs.tolist(),
        "tgt_attrs": tgt_attrs.tolist(),
        "instruction_text": sample_payload.get("instruction_text"),
        "condition_mode": args.condition_mode,
        "flip_beta_sign": bool(args.flip_beta_sign),
        "seed": int(args.seed),
        "task_id": None if args.task_id < 0 else int(args.task_id),
        "rows": rows,
        "summary": summary,
        "scalar_diagnostics": scalar_diagnostics,
        "diff_0_1_linf": float(np.max(np.abs(outputs[0] - outputs[1]))),
        "diff_1_2_linf": float(np.max(np.abs(outputs[1] - outputs[2]))),
        "diff_0_2_linf": float(np.max(np.abs(outputs[0] - outputs[2]))),
        "diagnostics": _to_jsonable(diagnostics),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
