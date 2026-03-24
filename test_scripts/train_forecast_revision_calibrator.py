from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.edit_spec_learned import fit_linear_calibrator, save_model
from modules.forecast_revision import extract_gt_edit_spec, search_teacher_edit_spec


def _gt_region(sample: dict) -> list[int]:
    mask = sample.get("edit_mask_gt", [])
    idx = [i for i, value in enumerate(mask) if value > 0]
    if not idx:
        return [0, 0]
    return [idx[0], idx[-1] + 1]


def _model_filename(model_kind: str) -> str:
    return {
        "linear": "learned_linear_calibrator.json",
        "family_affine": "learned_family_affine_calibrator.json",
        "family_duration_affine": "learned_family_duration_affine_calibrator.json",
    }[model_kind]


def _build_group_coverage(train_samples: list[dict], model_kind: str) -> dict:
    effect_counts: dict[str, int] = {}
    family_duration_counts: dict[str, int] = {}
    for sample in train_samples:
        intent = dict(sample.get("intent") or {})
        effect = str(intent.get("effect_family", "none"))
        duration = str(intent.get("duration", "none"))
        effect_counts[effect] = effect_counts.get(effect, 0) + 1
        key = f"{effect}::{duration}"
        family_duration_counts[key] = family_duration_counts.get(key, 0) + 1

    payload = {
        "model_kind": model_kind,
        "effect_family_counts": effect_counts,
        "effect_family_active_groups": sorted(key for key, count in effect_counts.items() if count >= 4),
        "effect_family_skipped_groups": sorted(key for key, count in effect_counts.items() if count < 4),
    }
    if model_kind == "family_duration_affine":
        payload["effect_family_duration_counts"] = family_duration_counts
        payload["effect_family_duration_active_groups"] = sorted(key for key, count in family_duration_counts.items() if count >= 4)
        payload["effect_family_duration_skipped_groups"] = sorted(key for key, count in family_duration_counts.items() if count < 4)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight learned edit-spec calibrator.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--label-source", choices=["gt", "teacher_search"], default="gt")
    parser.add_argument("--model-kind", choices=["linear", "family_affine", "family_duration_affine"], default="linear")
    args = parser.parse_args()

    payload = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    samples = list(payload.get("samples", []))
    if len(samples) < 4:
        raise ValueError("Need at least 4 samples to train/test the calibrator")

    rng = random.Random(args.seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    split = max(1, min(len(samples) - 1, int(round(len(samples) * args.train_ratio))))
    train_idx = set(indices[:split])

    train_samples = []
    heldout_samples = []
    teacher_debug = []
    for i, sample in enumerate(samples):
        enriched = dict(sample)
        enriched["intent"] = dict(sample.get("edit_intent_gt") or {})
        enriched["region"] = _gt_region(sample)
        if args.label_source == "teacher_search":
            teacher_spec, teacher_meta = search_teacher_edit_spec(
                intent=enriched["intent"],
                region=enriched["region"],
                history_ts=np.asarray(sample["history_ts"], dtype=np.float64),
                base_forecast=np.asarray(sample["base_forecast"], dtype=np.float64),
                revision_target=np.asarray(sample["revision_target"], dtype=np.float64),
                future_gt=np.asarray(sample["future_gt"], dtype=np.float64),
                gt_mask=np.asarray(sample["edit_mask_gt"], dtype=np.float64),
                context_text=str(sample.get("context_text", "")),
            )
            enriched["edit_spec_gt"] = teacher_spec
            enriched["teacher_search_meta"] = teacher_meta
            teacher_debug.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "teacher_spec": teacher_spec,
                    "teacher_meta": teacher_meta,
                }
            )
        else:
            enriched["edit_spec_gt"] = sample.get("edit_spec_gt") or extract_gt_edit_spec(sample)
        if i in train_idx:
            train_samples.append(enriched)
        else:
            heldout_samples.append(sample)

    model = fit_linear_calibrator(train_samples, alpha=args.alpha, model_kind=args.model_kind)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    model_path = out_root / _model_filename(args.model_kind)
    save_model(model, str(model_path))

    heldout_payload = dict(payload)
    heldout_payload["samples"] = heldout_samples
    heldout_path = out_root / "heldout_benchmark.json"
    heldout_path.write_text(json.dumps(heldout_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    train_payload = dict(payload)
    train_payload["samples"] = train_samples
    train_path = out_root / "train_split_debug.json"
    train_path.write_text(json.dumps(train_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if teacher_debug:
        teacher_path = out_root / "teacher_search_debug.json"
        teacher_path.write_text(json.dumps(teacher_debug, ensure_ascii=False, indent=2), encoding="utf-8")

    coverage_path = out_root / "group_coverage.json"
    coverage_path.write_text(
        json.dumps(_build_group_coverage(train_samples, args.model_kind), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({
        "benchmark": args.benchmark,
        "train_count": len(train_samples),
        "heldout_count": len(heldout_samples),
        "model_path": str(model_path),
        "heldout_benchmark": str(heldout_path),
        "train_split_debug": str(train_path),
        "alpha": args.alpha,
        "seed": args.seed,
        "label_source": args.label_source,
        "model_kind": args.model_kind,
        "group_coverage": str(coverage_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
