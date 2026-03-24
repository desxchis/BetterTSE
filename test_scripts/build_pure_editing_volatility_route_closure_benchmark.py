from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.pure_editing_volatility import classify_volatility_subpattern
from test_scripts.build_event_driven_testset import CSVDataLoader, EventDrivenSample, InjectorFactory
from test_scripts.build_pure_editing_how_much_stress_benchmark import (
    _duration_from_bucket,
    _start_step_for_duration,
    _strength_bucket,
    _target_energy_type,
)


TARGET_SUBPATTERNS = (
    "uniform_variance",
    "local_burst",
    "monotonic_envelope",
    "non_monotonic_envelope",
)


def _route_prompt(feature_desc: str, subpattern: str, duration_bucket: str, rng: np.random.RandomState) -> str:
    if subpattern == "uniform_variance":
        return f"请把{feature_desc}在一个{duration_bucket}窗口里整体变得更不稳定、整段更噪一些，只改那一段。"
    if subpattern == "local_burst":
        return f"请把{feature_desc}在一个{duration_bucket}窗口里只在局部短时片段出现突发式杂乱波动，其他位置尽量不动。"
    if subpattern == "monotonic_envelope":
        if rng.rand() < 0.5:
            return f"请把{feature_desc}在一个{duration_bucket}窗口里做成波动逐渐加剧、越来越乱的局部噪声变化。"
        return f"请把{feature_desc}在一个{duration_bucket}窗口里做成先更乱后逐步恢复、单调减弱的局部噪声变化。"
    return f"请把{feature_desc}在一个{duration_bucket}窗口里做成波动强度先增强后减弱再增强的多段变化，属于预览型复杂噪声。"


def build_route_closure_benchmark(
    *,
    csv_path: str,
    dataset_name: str,
    output_dir: str,
    per_subpattern: int,
    seq_len: int,
    random_seed: int,
    max_attempts: int,
) -> Dict[str, Any]:
    rng = np.random.RandomState(random_seed)
    data_loader = CSVDataLoader(csv_path, dataset_name)
    injector_factory = InjectorFactory(random_seed)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    quotas = {name: int(per_subpattern) for name in TARGET_SUBPATTERNS}
    counts: Counter[str] = Counter()
    samples: List[Dict[str, Any]] = []
    attempts = 0
    sample_idx = 1

    while any(counts[name] < quotas[name] for name in TARGET_SUBPATTERNS):
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"could not satisfy route closure quotas after {max_attempts} attempts: "
                f"current_counts={dict(counts)} target={quotas}"
            )

        duration_bucket = ["short", "medium", "long"][attempts % 3]
        feature = str(rng.choice(data_loader.features))
        start_idx = int(rng.randint(0, max(1, len(data_loader.data) - seq_len - 1)))
        base_ts, _ = data_loader.get_sequence(start_idx, seq_len, feature)
        injector = injector_factory.create_injector("noise_injection")
        duration = _duration_from_bucket(seq_len, duration_bucket, rng)
        start_step = _start_step_for_duration(seq_len, duration, rng)
        target_ts, mask_gt, injection_config = injector.inject(base_ts, start_step, duration)
        end_step = int(injection_config["end_step"])
        target_region = np.asarray(target_ts[start_step : end_step + 1], dtype=np.float32)
        base_region = np.asarray(base_ts[start_step : end_step + 1], dtype=np.float32)
        subpattern = classify_volatility_subpattern(target_region, base_region)
        if subpattern not in quotas or counts[subpattern] >= quotas[subpattern]:
            continue

        edit_intent_gt = injector.get_edit_intent(injection_config)
        edit_intent_gt["duration"] = duration_bucket
        edit_intent_gt["strength"] = _strength_bucket(
            injection_type="noise_injection",
            injection_config=injection_config,
            base_ts=base_ts,
        )
        feature_desc = data_loader.feature_descriptions.get(feature, injector_factory.get_feature_description(feature))
        prompt_text = _route_prompt(feature_desc, subpattern, duration_bucket, rng)
        sample = EventDrivenSample(
            sample_id=f"volroute_{sample_idx:03d}",
            dataset_name=dataset_name,
            target_feature=feature,
            feature_description=feature_desc,
            task_type=injector.get_task_type(),
            legacy_task_type=injector.get_legacy_task_type(),
            injection_operator=injector.get_injection_operator(),
            edit_intent_gt=edit_intent_gt,
            gt_start=int(injection_config["start_step"]),
            gt_end=end_step,
            event_prompts=[
                {
                    "level": 0,
                    "level_name": "volatility_route_closure",
                    "perspective": "protocol",
                    "prompt": prompt_text,
                }
            ],
            technical_ground_truth=f"[Volatility Route Closure] noise_injection {duration_bucket} {subpattern}",
            base_ts=np.asarray(base_ts, dtype=float).tolist(),
            target_ts=np.asarray(target_ts, dtype=float).tolist(),
            mask_gt=np.asarray(mask_gt, dtype=int).tolist(),
            injection_config=injection_config,
            causal_scenario=f"volatility_route_closure::{subpattern}::{duration_bucket}",
            seq_len=seq_len,
            timestamp=datetime.now().isoformat(),
        )
        sample_payload = asdict(sample)
        sample_payload["vague_prompt"] = prompt_text
        sample_payload["stress_metadata"] = {
            "duration_bucket": duration_bucket,
            "strength_bucket": edit_intent_gt["strength"],
            "target_energy_type": _target_energy_type(base_ts, target_ts, sample.gt_start, sample.gt_end),
            "injection_type": "noise_injection",
            "volatility_subpattern": subpattern,
            "benchmark_mode": "route_closure",
        }
        samples.append(sample_payload)
        counts[subpattern] += 1
        sample_idx += 1

    output_payload = {
        "benchmark_type": "pure_editing_volatility_route_closure",
        "dataset_name": dataset_name,
        "seq_len": seq_len,
        "num_samples": len(samples),
        "random_seed": random_seed,
        "per_subpattern": per_subpattern,
        "target_subpatterns": list(TARGET_SUBPATTERNS),
        "collected_counts": dict(counts),
        "attempts": attempts,
        "samples": samples,
    }
    json_path = output_root / f"pure_editing_volatility_route_closure_{dataset_name}_{len(samples)}.json"
    json_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = output_root / "README.md"
    summary_lines = [
        "# Pure Editing Volatility Route Closure Benchmark",
        "",
        f"- dataset_name: {dataset_name}",
        f"- num_samples: {len(samples)}",
        f"- seq_len: {seq_len}",
        f"- random_seed: {random_seed}",
        f"- attempts: {attempts}",
        "",
        "## Collected Counts",
        "",
    ]
    for name in TARGET_SUBPATTERNS:
        summary_lines.append(f"- `{name}`: {counts[name]}")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {"json_path": str(json_path), "summary_path": str(summary_path), "num_samples": len(samples), "attempts": attempts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a subtype-aware route closure benchmark for volatility routing.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--dataset-name", default="ETTh1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-subpattern", type=int, default=6)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--random-seed", type=int, default=37)
    parser.add_argument("--max-attempts", type=int, default=8000)
    args = parser.parse_args()

    result = build_route_closure_benchmark(
        csv_path=args.csv_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        per_subpattern=args.per_subpattern,
        seq_len=args.seq_len,
        random_seed=args.random_seed,
        max_attempts=args.max_attempts,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
