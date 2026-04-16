from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.edit_spec_learned import load_model as load_calibrator_model, predict_with_model
from modules.forecast_revision import (
    apply_revision_profile,
    compute_intent_alignment,
    evaluate_revision_sample,
    heuristic_revision_plan,
    predict_edit_spec,
    project_edit_spec_to_params,
)
from modules.reliability_learned import fit_linear_reliability_model, save_model


def _disagreement(rule: dict, learned: dict) -> dict[str, float]:
    return {
        "delta_gap": abs(float(learned.get("delta_level_z", 0.0)) - float(rule.get("delta_level_z", 0.0))),
        "duration_gap": abs(float(learned.get("duration_ratio", 0.0)) - float(rule.get("duration_ratio", 0.0))),
        "amp_gap": abs(float(learned.get("amp_ratio", 0.0)) - float(rule.get("amp_ratio", 0.0))),
        "slope_gap": abs(float(learned.get("slope_ratio", 0.0)) - float(rule.get("slope_ratio", 0.0))),
        "recovery_gap": abs(float(learned.get("recovery_ratio", 0.0)) - float(rule.get("recovery_ratio", 0.0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight semantic reliability estimator for learned calibration gating.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--learned-calibrator", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--target-mode", choices=["semantic", "benefit", "hybrid"], default="semantic")
    parser.add_argument("--benefit-margin", type=float, default=0.0)
    args = parser.parse_args()

    payload = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    samples = list(payload.get("samples", []))
    calibrator_model = load_calibrator_model(args.learned_calibrator)

    train_rows = []
    for sample in samples:
        plan = heuristic_revision_plan(sample["context_text"], sample["forecast_horizon"])
        if not plan.get("revision_needed", False):
            continue

        history_ts = np.asarray(sample["history_ts"], dtype=np.float64)
        base_forecast = np.asarray(sample["base_forecast"], dtype=np.float64)
        region = list(plan["localization"]["region"])
        intent = dict(plan["intent"])

        rule_spec = predict_edit_spec(
            intent=intent,
            region=region,
            history_ts=history_ts,
            base_forecast=base_forecast,
            context_text=sample["context_text"],
            strategy="rule_local_stats",
        )
        learned_spec = predict_with_model(
            calibrator_model,
            intent=intent,
            region=region,
            history_ts=history_ts,
            base_forecast=base_forecast,
            context_text=sample["context_text"],
        )
        alignment = compute_intent_alignment(plan, sample)
        rule_params = project_edit_spec_to_params(
            edit_spec=rule_spec,
            intent=intent,
            region=region,
            history_ts=history_ts,
            base_forecast=base_forecast,
        )
        learned_params = project_edit_spec_to_params(
            edit_spec=learned_spec,
            intent=intent,
            region=region,
            history_ts=history_ts,
            base_forecast=base_forecast,
        )
        rule_edited, _ = apply_revision_profile(base_forecast, intent, region, rule_params)
        learned_edited, _ = apply_revision_profile(base_forecast, intent, region, learned_params)
        metrics_rule = evaluate_revision_sample(
            base_forecast=base_forecast,
            edited_forecast=rule_edited,
            future_gt=np.asarray(sample["future_gt"], dtype=np.float64),
            revision_target=np.asarray(sample["revision_target"], dtype=np.float64),
            pred_region=region,
            gt_mask=np.asarray(sample["edit_mask_gt"], dtype=np.float64),
        )
        metrics_learned = evaluate_revision_sample(
            base_forecast=base_forecast,
            edited_forecast=learned_edited,
            future_gt=np.asarray(sample["future_gt"], dtype=np.float64),
            revision_target=np.asarray(sample["revision_target"], dtype=np.float64),
            pred_region=region,
            gt_mask=np.asarray(sample["edit_mask_gt"], dtype=np.float64),
        )
        semantic_target = float(alignment["intent_match_score"])
        benefit_target = 1.0 if (metrics_learned["revision_gain"] >= metrics_rule["revision_gain"] + args.benefit_margin) else 0.0
        if args.target_mode == "semantic":
            target_value = semantic_target
        elif args.target_mode == "benefit":
            target_value = benefit_target
        else:
            target_value = 0.5 * semantic_target + 0.5 * benefit_target
        train_rows.append({
            "sample_id": sample["sample_id"],
            "history_ts": sample["history_ts"],
            "base_forecast": sample["base_forecast"],
            "context_text": sample["context_text"],
            "intent": intent,
            "region": region,
            "tool_name": plan.get("tool_name", "none"),
            "plan_confidence": float(plan.get("confidence", 0.75)),
            "reliability_target": float(target_value),
            "semantic_target": semantic_target,
            "benefit_target": benefit_target,
            "rule_revision_gain": float(metrics_rule["revision_gain"]),
            "learned_revision_gain": float(metrics_learned["revision_gain"]),
            "disagreement_features": _disagreement(rule_spec, learned_spec),
        })

    model = fit_linear_reliability_model(train_rows, alpha=args.alpha)
    bundle = {
        "bundle_type": "learned_reliability_gate",
        "threshold": float(args.threshold),
        "learned_calibrator_path": str(Path(args.learned_calibrator).resolve()),
        "reliability_model": model,
        "train_count": len(train_rows),
        "benchmark": args.benchmark,
        "target_mode": args.target_mode,
        "benefit_margin": float(args.benefit_margin),
    }
    save_model(bundle, args.output_path)

    print(json.dumps({
        "benchmark": args.benchmark,
        "learned_calibrator": args.learned_calibrator,
        "output_path": args.output_path,
        "train_count": len(train_rows),
        "alpha": args.alpha,
        "threshold": args.threshold,
        "target_mode": args.target_mode,
        "benefit_margin": args.benefit_margin,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
