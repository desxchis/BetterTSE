from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
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


SUBTYPE_BY_SUBPATTERN = {
    "uniform_variance": "global_scale",
    "local_burst": "local_burst",
    "monotonic_envelope": "envelope_monotonic",
    "non_monotonic_envelope": "preview_non_monotonic",
}


def _bucket_from_region(start: int, end: int, seq_len: int) -> str:
    center = 0.5 * (int(start) + int(end))
    if center < seq_len / 3:
        return "early"
    if center < 2 * seq_len / 3:
        return "mid"
    return "late"


def _time_phrases(bucket: str) -> Dict[int, str]:
    mapping = {
        "early": {
            1: "在今日运行前段",
            2: "从今天早些时候开始",
            3: "前面那一段时间",
        },
        "mid": {
            1: "在今日运行中段",
            2: "从今天中午开始",
            3: "刚才那一阵",
        },
        "late": {
            1: "在夜间低谷期前",
            2: "预计在今晚深夜",
            3: "就快到半夜的时候",
        },
    }
    return mapping[bucket]


def _feature_label(sample: Dict[str, Any]) -> str:
    return str(sample.get("feature_description") or sample.get("target_feature") or "相关监测信号")


def _volatility_prompt(feature_desc: str, subtype: str, level: int, bucket: str) -> str:
    time_phrase = _time_phrases(bucket)[level]
    if subtype == "global_scale":
        prompts = {
            1: f"请注意，{time_phrase}，{feature_desc}读数会整体更不稳定、整段噪声抬升，请重点核查该段监测链路。",
            2: f"监测部门通报，{time_phrase}，{feature_desc}监测信号将呈现整段更噪、整体波动增强的失真表现。",
            3: f"{time_phrase}，这路信号看着不是局部乱跳，而是整段都更噪、更不稳定了，像是整体失真在抬高。",
        }
        return prompts[level]
    if subtype == "local_burst":
        prompts = {
            1: f"请注意，{time_phrase}，{feature_desc}读数会在局部短时片段出现突发式杂乱波动，请重点排查那一小段。",
            2: f"监测部门通报，{time_phrase}，{feature_desc}监测信号将在局部时段出现突发式杂乱波动，而非整段同步失真。",
            3: f"{time_phrase}，这路信号像是某一小段突然炸了一下，局部短时片段特别乱，别的地方倒没一起坏掉。",
        }
        return prompts[level]
    if subtype == "envelope_monotonic":
        prompts = {
            1: f"请注意，{time_phrase}，{feature_desc}读数会呈现波动逐渐加剧或逐步恢复的单调噪声变化，请关注整段演化方向。",
            2: f"监测部门通报，{time_phrase}，{feature_desc}监测信号将表现为波动逐步加剧或逐步减弱的单调失真过程。",
            3: f"{time_phrase}，这路信号不是一下子炸掉，而是越往后越乱，或者慢慢恢复，整体是单调地变坏或变稳。",
        }
        return prompts[level]
    prompts = {
        1: f"请注意，{time_phrase}，{feature_desc}读数会出现先增强后减弱再增强的多段复杂失真，这类情况先按预览型复杂噪声处理。",
        2: f"监测部门通报，{time_phrase}，{feature_desc}监测信号将呈现先增强后减弱再增强的多段变化，属于预览型复杂噪声。",
        3: f"{time_phrase}，这路信号不是单调变坏，而是先高后低再高、反复起伏，像多段变化那种复杂失真，先当预览难例看。",
    }
    return prompts[level]


def _refresh_noise_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    refreshed_sample = deepcopy(sample)
    base_ts = np.asarray(refreshed_sample["base_ts"], dtype=np.float32)
    target_ts = np.asarray(refreshed_sample["target_ts"], dtype=np.float32)
    start = int(refreshed_sample["gt_start"])
    end = int(refreshed_sample["gt_end"]) + 1
    subpattern = classify_volatility_subpattern(target_ts[start:end], base_ts[start:end])
    subtype = SUBTYPE_BY_SUBPATTERN[subpattern]
    bucket = _bucket_from_region(start, end, int(refreshed_sample.get("seq_len", len(base_ts))))
    feature_desc = _feature_label(refreshed_sample)

    refreshed_sample["original_event_prompts"] = deepcopy(refreshed_sample.get("event_prompts", []))
    refreshed_sample["volatility_subpattern"] = subpattern
    refreshed_sample["volatility_subtype_gt"] = subtype
    refreshed_sample.setdefault("stress_metadata", {})
    refreshed_sample["stress_metadata"]["volatility_subpattern"] = subpattern
    refreshed_sample["stress_metadata"]["volatility_subtype_gt"] = subtype

    edit_intent_gt = refreshed_sample.get("edit_intent_gt") if isinstance(refreshed_sample.get("edit_intent_gt"), dict) else {}
    edit_intent_gt["volatility_subtype"] = subtype
    refreshed_sample["edit_intent_gt"] = edit_intent_gt

    refreshed_sample["event_prompts"] = [
        {
            "level": level,
            "level_name": level_name,
            "perspective": perspective,
            "prompt": _volatility_prompt(feature_desc, subtype, level, bucket),
        }
        for level, level_name, perspective in [
            (1, "直接业务指令", "调度员视角"),
            (2, "宏观新闻播报", "新闻主播视角"),
            (3, "无关联线索", "社交媒体路人视角"),
        ]
    ]
    refreshed_sample["vague_prompt"] = refreshed_sample["event_prompts"][1]["prompt"]
    refreshed_sample["technical_ground_truth"] = (
        str(refreshed_sample.get("technical_ground_truth", ""))
        + f"\nVolatility Subpattern: {subpattern}\nVolatility Subtype GT: {subtype}"
    )
    return refreshed_sample


def _build_monotonic_noise_samples(
    *,
    csv_path: str,
    dataset_name: str,
    seq_len: int,
    random_seed: int,
    needed_count: int,
    start_sample_index: int,
    max_attempts: int,
) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(random_seed)
    data_loader = CSVDataLoader(csv_path, dataset_name)
    injector_factory = InjectorFactory(random_seed)
    monotonic_samples: List[Dict[str, Any]] = []
    attempts = 0

    while len(monotonic_samples) < needed_count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"could not collect {needed_count} monotonic_envelope samples after {max_attempts} attempts"
            )
        feature = str(rng.choice(data_loader.features))
        start_idx = int(rng.randint(0, max(1, len(data_loader.data) - seq_len - 1)))
        base_ts, _ = data_loader.get_sequence(start_idx, seq_len, feature)
        injector = injector_factory.create_injector("noise_injection")
        duration_bucket = ["short", "medium", "long"][attempts % 3]
        duration = _duration_from_bucket(seq_len, duration_bucket, rng)
        start_step = _start_step_for_duration(seq_len, duration, rng)
        target_ts, mask_gt, injection_config = injector.inject(base_ts, start_step, duration)
        end_step = int(injection_config["end_step"])
        subpattern = classify_volatility_subpattern(
            np.asarray(target_ts[start_step : end_step + 1], dtype=np.float32),
            np.asarray(base_ts[start_step : end_step + 1], dtype=np.float32),
        )
        if subpattern != "monotonic_envelope":
            continue

        feature_desc = data_loader.feature_descriptions.get(feature, injector_factory.get_feature_description(feature))
        edit_intent_gt = injector.get_edit_intent(injection_config)
        edit_intent_gt["duration"] = duration_bucket
        edit_intent_gt["strength"] = _strength_bucket(
            injection_type="noise_injection",
            injection_config=injection_config,
            base_ts=base_ts,
        )
        sample = EventDrivenSample(
            sample_id=f"{start_sample_index + len(monotonic_samples):03d}",
            dataset_name=dataset_name,
            target_feature=feature,
            feature_description=feature_desc,
            task_type=injector.get_task_type(),
            legacy_task_type=injector.get_legacy_task_type(),
            injection_operator=injector.get_injection_operator(),
            edit_intent_gt=edit_intent_gt,
            gt_start=int(injection_config["start_step"]),
            gt_end=end_step,
            event_prompts=[],
            technical_ground_truth=f"[Mainline VolSubtype v2] monotonic_envelope noise_injection {duration_bucket}",
            base_ts=np.asarray(base_ts, dtype=float).tolist(),
            target_ts=np.asarray(target_ts, dtype=float).tolist(),
            mask_gt=np.asarray(mask_gt, dtype=int).tolist(),
            injection_config=injection_config,
            causal_scenario=f"volatility_monotonic_extension::{duration_bucket}",
            seq_len=seq_len,
            timestamp=datetime.now().isoformat(),
        )
        payload = asdict(sample)
        payload.setdefault("stress_metadata", {})
        payload["stress_metadata"]["duration_bucket"] = duration_bucket
        payload["stress_metadata"]["strength_bucket"] = edit_intent_gt["strength"]
        payload["stress_metadata"]["target_energy_type"] = _target_energy_type(base_ts, target_ts, sample.gt_start, sample.gt_end)
        payload["stress_metadata"]["injection_type"] = "noise_injection"
        monotonic_samples.append(_refresh_noise_sample(payload))

    return monotonic_samples


def refresh_benchmark(
    *,
    benchmark_path: str,
    output_dir: str,
    output_name: str | None = None,
    csv_path: str | None = None,
    target_monotonic_count: int = 0,
    random_seed: int = 41,
    max_attempts: int = 8000,
) -> Dict[str, Any]:
    source = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    refreshed = deepcopy(source)
    refreshed.setdefault("metadata", {})
    refreshed["metadata"]["volatility_subtype_aware_v2"] = True
    refreshed["metadata"]["source_benchmark"] = str(benchmark_path)

    subtype_counts: Dict[str, int] = {key: 0 for key in SUBTYPE_BY_SUBPATTERN.values()}
    changed = 0
    for sample in refreshed.get("samples", []):
        if sample.get("injection_operator") != "noise_injection":
            continue
        updated = _refresh_noise_sample(sample)
        sample.update(updated)
        subtype_counts[updated["volatility_subtype_gt"]] += 1
        changed += 1

    if target_monotonic_count > subtype_counts["envelope_monotonic"]:
        if not csv_path:
            raise ValueError("csv_path is required when target_monotonic_count exceeds current monotonic coverage")
        needed = target_monotonic_count - subtype_counts["envelope_monotonic"]
        next_index = max(int(str(sample.get("sample_id", "0"))) for sample in refreshed.get("samples", []) if str(sample.get("sample_id", "")).isdigit()) + 1
        seq_len = int(refreshed.get("metadata", {}).get("seq_len", 192))
        dataset_name = str(refreshed.get("metadata", {}).get("dataset_name", "ETTh1"))
        supplement = _build_monotonic_noise_samples(
            csv_path=csv_path,
            dataset_name=dataset_name,
            seq_len=seq_len,
            random_seed=random_seed,
            needed_count=needed,
            start_sample_index=next_index,
            max_attempts=max_attempts,
        )
        refreshed["samples"].extend(supplement)
        subtype_counts["envelope_monotonic"] += len(supplement)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        src_name = Path(benchmark_path).stem
        output_name = f"{src_name}_volsubtype_v2.json"
    output_path = output_root / output_name
    output_path.write_text(json.dumps(refreshed, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = output_root / "README.md"
    summary_lines = [
        "# Event-Driven Volatility Subtype-Aware Mainline v2",
        "",
        f"- source_benchmark: {benchmark_path}",
        f"- changed_volatility_samples: {changed}",
        "",
        "## Volatility Subtype Counts",
        "",
    ]
    for subtype, count in subtype_counts.items():
        summary_lines.append(f"- `{subtype}`: {count}")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "changed_volatility_samples": changed,
        "subtype_counts": subtype_counts,
        "total_samples": len(refreshed.get("samples", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh an event-driven mainline benchmark with volatility subtype-aware prompts and labels.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--target-monotonic-count", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=41)
    parser.add_argument("--max-attempts", type=int, default=8000)
    args = parser.parse_args()

    result = refresh_benchmark(
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
        output_name=args.output_name,
        csv_path=args.csv_path,
        target_monotonic_count=args.target_monotonic_count,
        random_seed=args.random_seed,
        max_attempts=args.max_attempts,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
