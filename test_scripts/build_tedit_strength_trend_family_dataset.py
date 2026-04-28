from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from test_scripts.build_tedit_strength_discrete_benchmark import build_discrete_benchmark


DEFAULT_STRENGTH_TO_SCALAR = {
    "weak": 0.0,
    "medium": 0.5,
    "strong": 1.0,
}

LEGACY_0_1_2_STRENGTH_TO_SCALAR = {
    "weak": 0.0,
    "medium": 1.0,
    "strong": 2.0,
}

SCALAR_SCHEME_TO_AXIS = {
    "default_0_0p5_1": {
        "anchor_mapping": DEFAULT_STRENGTH_TO_SCALAR,
        "range": [0.0, 1.0],
    },
    "legacy_0_1_2": {
        "anchor_mapping": LEGACY_0_1_2_STRENGTH_TO_SCALAR,
        "range": [0.0, 2.0],
    },
}


SELECTOR_CONTROL_META = {
    "trend_injection": {
        "control_attr": ["trend_types"],
        "control_attr_ids": [0],
    },
    "seasonality_injection": {
        "control_attr": ["season_cycles"],
        "control_attr_ids": [2],
    },
}


def build_family_dataset(
    *,
    csv_path: str,
    dataset_name: str,
    output_dir: str,
    seq_len: int,
    random_seed: int,
    train_families: int,
    valid_families: int,
    test_families: int,
    injection_types: list[str] | None = None,
    selector: str | None = None,
    collection_root: str | None = None,
    scalar_scheme: str = "default_0_0p5_1",
) -> dict[str, object]:
    chosen_injection_types = list(injection_types or ["trend_injection"])
    resolved_selector = str(selector or chosen_injection_types[0]).strip()
    base_root = Path(collection_root) if collection_root else Path(output_dir)
    output_root = base_root / resolved_selector if collection_root else Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if not resolved_selector:
        raise ValueError("selector must be non-empty")
    if len(chosen_injection_types) != 1:
        raise ValueError("Dedicated family dataset build expects exactly one injection type")
    if chosen_injection_types[0] != resolved_selector:
        raise ValueError(f"selector '{resolved_selector}' must match the dedicated injection type '{chosen_injection_types[0]}'")
    axis_spec = SCALAR_SCHEME_TO_AXIS.get(str(scalar_scheme).strip())
    if axis_spec is None:
        raise ValueError(f"Unsupported scalar_scheme={scalar_scheme}")
    strength_to_scalar = dict(axis_spec["anchor_mapping"])

    split_specs = [
        ("train", train_families, random_seed),
        ("valid", valid_families, random_seed + 101),
        ("test", test_families, random_seed + 202),
    ]

    split_results: dict[str, dict[str, object]] = {}
    for split, num_families, seed in split_specs:
        result = build_discrete_benchmark(
            csv_path=csv_path,
            dataset_name=dataset_name,
            output_dir=str(output_root / split),
            num_families=num_families,
            seq_len=seq_len,
            random_seed=seed,
            injection_types=chosen_injection_types,
        )
        src_json = Path(result["json_path"])
        payload = json.loads(src_json.read_text(encoding="utf-8"))
        for family in payload.get("families", []):
            samples = list(family.get("samples", []))
            samples.sort(key=lambda sample: float(sample.get("strength_scalar", strength_to_scalar.get(str(sample.get("strength_text", "")), 0.0))))
            for sample in samples:
                strength_text = str(sample.get("strength_text", ""))
                strength_scalar = float(strength_to_scalar.get(strength_text, sample.get("strength_scalar", 0.0)))
                sample["strength_scalar"] = strength_scalar
            family["samples"] = samples
        payload["scalar_scheme"] = str(scalar_scheme)
        payload["strength_axis"] = {
            "type": "continuous_scalar",
            "anchor_mapping": strength_to_scalar,
            "range": list(axis_spec["range"]),
        }
        (output_root / f"{split}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        split_results[split] = {
            "num_families": int(payload["num_families"]),
            "num_samples": int(payload["num_samples"]),
            "random_seed": int(seed),
        }

    selector_meta = SELECTOR_CONTROL_META.get(
        resolved_selector,
        {
            "control_attr": [],
            "control_attr_ids": [],
        },
    )

    meta = {
        "dataset_type": "discrete_strength_family",
        "dataset_name": dataset_name,
        "seq_len": seq_len,
        "attr_list": ["trend_types", "trend_directions", "season_cycles"],
        "attr_n_ops": [4, 2, 4],
        "control_attr": selector_meta["control_attr"],
        "control_attr_ids": selector_meta["control_attr_ids"],
        "strength_bins": ["weak", "medium", "strong"],
        "scalar_scheme": str(scalar_scheme),
        "selector": resolved_selector,
        "injection_types": chosen_injection_types,
        "semantic_alignment": {
            "mode": "unified_strength_scalar_plus_family_mapping",
            "semantic_authority": "benchmark_family_records",
            "use_text_context": True,
            "use_task_id": False,
            "step_change_default_attr_strategy": "neutral",
        },
        "strength_axis": {
            "type": "continuous_scalar",
            "anchor_mapping": strength_to_scalar,
            "range": list(axis_spec["range"]),
        },
        "splits": split_results,
    }
    (output_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "README.md").write_text(
        "\n".join(
            [
                "# Discrete Strength Family Dataset",
                "",
                f"- dataset_name: {dataset_name}",
                f"- seq_len: {seq_len}",
                f"- train_families: {train_families}",
                f"- valid_families: {valid_families}",
                f"- test_families: {test_families}",
                f"- selector: {resolved_selector}",
                f"- injection_types: {', '.join(chosen_injection_types)}",
                "- semantic_mode: unified_strength_scalar_plus_family_mapping",
                "- semantic_authority: benchmark family-level fields (tool_name/effect_family/shape/direction/task_id/instruction_text/attr_strategy)",
                "- task_gate_default: off (use_task_id=false)",
                "",
                f"每个 family 固定 source/region/template，只改变有序 strength scalar（scheme={scalar_scheme}；anchor: weak={strength_to_scalar['weak']}, medium={strength_to_scalar['medium']}, strong={strength_to_scalar['strong']}）。",
                "非 trend/seasonality family 的主语义由 instruction_text 承担，task_id 仅作为受控 gate 的辅助语义通道。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "collection_root": str(base_root if collection_root else output_root.parent),
        "selector": resolved_selector,
        "output_dir": str(output_root),
        "splits": split_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a discrete strength family dataset for TEdit training.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--dataset-name", default="ETTh1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--collection-root", default="")
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--train-families", type=int, default=96)
    parser.add_argument("--valid-families", type=int, default=24)
    parser.add_argument("--test-families", type=int, default=24)
    parser.add_argument("--injection-types", default="trend_injection")
    parser.add_argument("--selector", default="")
    parser.add_argument("--scalar-scheme", default="default_0_0p5_1", choices=sorted(SCALAR_SCHEME_TO_AXIS.keys()))
    args = parser.parse_args()

    injection_types = [token.strip() for token in args.injection_types.split(",") if token.strip()]
    selector = args.selector.strip() or (injection_types[0] if injection_types else "")

    result = build_family_dataset(
        csv_path=args.csv_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        random_seed=args.random_seed,
        train_families=args.train_families,
        valid_families=args.valid_families,
        test_families=args.test_families,
        injection_types=injection_types,
        selector=selector,
        collection_root=args.collection_root.strip() or None,
        scalar_scheme=str(args.scalar_scheme),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
