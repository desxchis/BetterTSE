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


def _extract_scalar_metrics(diagnostics: dict[str, object]) -> dict[str, object]:
    projector = diagnostics.get("projector") or []
    modulation_base = diagnostics.get("modulation_base") or []
    modulation_weaver = diagnostics.get("modulation_weaver") or []
    generator = diagnostics.get("generator") or []

    latest_generator = generator[-1] if generator else {}
    latest_projector = projector[-1] if projector else {}

    projector_pairwise = latest_projector.get("projector_output_pairwise_l2") or {}
    projector_strength_norms = latest_projector.get("projector_output_mean_norm_by_strength") or {}

    def _collect_modulation(records, key):
        values = []
        for record in records:
            value = record.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return _aggregate_mean(values)

    modulation_records = list(modulation_base) + list(modulation_weaver)
    return {
        "raw_edit_region_mean_abs_delta": latest_generator.get("raw_edit_region_mean_abs_delta"),
        "final_edit_region_mean_abs_delta": latest_generator.get("final_edit_region_mean_abs_delta"),
        "blend_gap_edit_region_mean_abs": latest_generator.get("blend_gap_edit_region_mean_abs"),
        "projector_pairwise_l2": projector_pairwise,
        "projector_strength_mean_norms": {
            str(k): v.get("norm") if isinstance(v, dict) else None
            for k, v in projector_strength_norms.items()
        },
        "modulation_delta_gamma_abs_mean": _collect_modulation(modulation_records, "delta_gamma_abs_mean"),
        "modulation_delta_beta_abs_mean": _collect_modulation(modulation_records, "delta_beta_abs_mean"),
        "modulation_delta_gamma_over_base_mean": _collect_modulation(modulation_records, "delta_gamma_over_base_mean"),
        "modulation_delta_beta_over_base_mean": _collect_modulation(modulation_records, "delta_beta_over_base_mean"),
    }


def _linf_diff(left: dict[str, object], right: dict[str, object], key: str) -> float | None:
    left_value = left.get(key)
    right_value = right.get(key)
    if left_value is None or right_value is None:
        return None
    return float(abs(float(left_value) - float(right_value)))


def _load_probe_sample(dataset_folder: Path, split: str, requested_idx: int) -> dict[str, object]:
    meta_path = dataset_folder / "meta.json"
    if meta_path.exists():
        from data.discrete_strength_family import DiscreteStrengthFamilyDataset

        dataset = DiscreteStrengthFamilyDataset(str(dataset_folder))
        split_ds = dataset.get_split(split, include_self=True)
        idx = int(requested_idx) if requested_idx >= 0 else 0
        family = split_ds[idx]
        sample = family["samples"][0]
        return {
            "probe_idx": idx,
            "base": np.asarray(sample["src_x"], dtype=np.float32).squeeze(-1),
            "src_attrs": np.asarray(sample["src_attrs"], dtype=np.int64),
            "tgt_attrs": np.asarray(sample["tgt_attrs"], dtype=np.int64),
            "instruction_text": str(sample.get("instruction_text")) if sample.get("instruction_text") is not None else None,
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
    }


def _select_sample(attrs: np.ndarray, requested_idx: int) -> int:
    if requested_idx >= 0:
        return int(requested_idx)
    for idx in range(len(attrs)):
        if not np.array_equal(attrs[idx, 0], attrs[idx, 1]):
            return idx
    raise ValueError("no non-self sample found in dataset")


def _resolve_instruction_text(dataset_folder: Path, split: str, idx: int) -> str | None:
    try:
        from data.synthetic_finetune import SyntheticDataset
    except Exception:
        return None
    dataset = SyntheticDataset(str(dataset_folder))
    sample = dataset.get_split(split, include_self=True)[idx]
    instruction_text = sample.get("instruction_text")
    if instruction_text is None:
        return None
    return str(instruction_text)


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
    parser.add_argument("--condition-mode", default="both", choices=["both", "label_only", "text_only"], help="ablate whether strength comes from numeric label, text, or both")
    parser.add_argument("--enable-strength-diagnostics", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_folder)
    sample_payload = _load_probe_sample(dataset_root, args.split, args.sample_idx)
    idx = int(sample_payload["probe_idx"])

    base = np.asarray(sample_payload["base"], dtype=np.float32)
    src_attrs = np.asarray(sample_payload["src_attrs"], dtype=np.int64)
    tgt_attrs = np.asarray(sample_payload["tgt_attrs"], dtype=np.int64)
    instruction_text = None
    if bool(args.use_dataset_instruction):
        instruction_text = sample_payload.get("instruction_text")
    if args.condition_mode == "label_only":
        instruction_text = None

    wrapper = TEditWrapper(model_path=args.model_path, config_path=args.config_path, device=args.device)

    rows = []
    outputs = []
    per_strength_diagnostics = {}
    for strength_label in [0, 1, 2]:
        run_strength_label = strength_label
        run_instruction_text = instruction_text
        if args.condition_mode == "text_only":
            run_strength_label = 0
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        edited, diagnostics = wrapper.edit_time_series(
            ts=base,
            src_attrs=src_attrs,
            tgt_attrs=tgt_attrs,
            n_samples=1,
            sampler="ddim",
            edit_steps=args.edit_steps,
            strength_label=run_strength_label,
            task_id=None if args.task_id < 0 else args.task_id,
            instruction_text=run_instruction_text,
            return_diagnostics=True,
            enable_strength_diagnostics=bool(args.enable_strength_diagnostics),
        )
        edited = edited[0]
        outputs.append(edited)
        per_strength_diagnostics[str(strength_label)] = diagnostics
        rows.append(
            {
                "strength_label": int(strength_label),
                "runtime_strength_label": int(run_strength_label),
                "peak_abs_delta": float(np.max(np.abs(edited - base))),
                "mean_abs_delta": float(np.mean(np.abs(edited - base))),
                "sum_delta": float(np.sum(edited - base)),
                "raw_edit_region_mean_abs_delta": diagnostics["generator"][0]["raw_edit_region_mean_abs_delta"] if diagnostics.get("generator") else None,
                "final_edit_region_mean_abs_delta": diagnostics["generator"][0]["final_edit_region_mean_abs_delta"] if diagnostics.get("generator") else None,
                "blend_gap_edit_region_mean_abs": diagnostics["generator"][0]["blend_gap_edit_region_mean_abs"] if diagnostics.get("generator") else None,
            }
        )

    strength_scalar_diagnostics = {
        str(label): _extract_scalar_metrics(diag)
        for label, diag in per_strength_diagnostics.items()
    }
    summary = {
        "condition_mode": args.condition_mode,
        "raw_edit_region_mean_abs_delta": {
            "weak": rows[0]["raw_edit_region_mean_abs_delta"],
            "medium": rows[1]["raw_edit_region_mean_abs_delta"],
            "strong": rows[2]["raw_edit_region_mean_abs_delta"],
        },
        "final_edit_region_mean_abs_delta": {
            "weak": rows[0]["final_edit_region_mean_abs_delta"],
            "medium": rows[1]["final_edit_region_mean_abs_delta"],
            "strong": rows[2]["final_edit_region_mean_abs_delta"],
        },
        "blend_gap_edit_region_mean_abs": {
            "weak": rows[0]["blend_gap_edit_region_mean_abs"],
            "medium": rows[1]["blend_gap_edit_region_mean_abs"],
            "strong": rows[2]["blend_gap_edit_region_mean_abs"],
        },
        "projector_pairwise_l2": {
            "weak": strength_scalar_diagnostics["0"].get("projector_pairwise_l2"),
            "medium": strength_scalar_diagnostics["1"].get("projector_pairwise_l2"),
            "strong": strength_scalar_diagnostics["2"].get("projector_pairwise_l2"),
        },
        "modulation_delta_gamma_abs_mean": {
            "weak": strength_scalar_diagnostics["0"].get("modulation_delta_gamma_abs_mean"),
            "medium": strength_scalar_diagnostics["1"].get("modulation_delta_gamma_abs_mean"),
            "strong": strength_scalar_diagnostics["2"].get("modulation_delta_gamma_abs_mean"),
        },
        "modulation_delta_beta_abs_mean": {
            "weak": strength_scalar_diagnostics["0"].get("modulation_delta_beta_abs_mean"),
            "medium": strength_scalar_diagnostics["1"].get("modulation_delta_beta_abs_mean"),
            "strong": strength_scalar_diagnostics["2"].get("modulation_delta_beta_abs_mean"),
        },
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
        "instruction_text": instruction_text,
        "condition_mode": args.condition_mode,
        "seed": int(args.seed),
        "task_id": None if args.task_id < 0 else int(args.task_id),
        "rows": rows,
        "summary": summary,
        "scalar_diagnostics": strength_scalar_diagnostics,
        "diff_0_1_linf": float(np.max(np.abs(outputs[0] - outputs[1]))),
        "diff_1_2_linf": float(np.max(np.abs(outputs[1] - outputs[2]))),
        "diff_0_2_linf": float(np.max(np.abs(outputs[0] - outputs[2]))),
        "diagnostics": _to_jsonable(per_strength_diagnostics),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
