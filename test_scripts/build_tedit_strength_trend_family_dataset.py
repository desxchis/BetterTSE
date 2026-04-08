from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from test_scripts.build_tedit_strength_discrete_benchmark import build_discrete_benchmark


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
) -> dict[str, object]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

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
            injection_types=["trend_injection"],
        )
        src_json = Path(result["json_path"])
        payload = json.loads(src_json.read_text(encoding="utf-8"))
        (output_root / f"{split}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        split_results[split] = {
            "num_families": int(payload["num_families"]),
            "num_samples": int(payload["num_samples"]),
            "random_seed": int(seed),
        }

    meta = {
        "dataset_type": "discrete_strength_family",
        "dataset_name": dataset_name,
        "seq_len": seq_len,
        "attr_list": ["trend_types", "trend_directions", "season_cycles"],
        "attr_n_ops": [4, 2, 4],
        "control_attr": ["trend_types"],
        "control_attr_ids": [0],
        "strength_bins": ["weak", "medium", "strong"],
        "splits": split_results,
    }
    (output_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "README.md").write_text(
        "\n".join(
            [
                "# Trend Strength Family Dataset",
                "",
                f"- dataset_name: {dataset_name}",
                f"- seq_len: {seq_len}",
                f"- train_families: {train_families}",
                f"- valid_families: {valid_families}",
                f"- test_families: {test_families}",
                "",
                "每个 family 固定 source/region/template，只改变 weak/medium/strong。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"output_dir": str(output_root), "splits": split_results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a trend-only family dataset for discrete strength TEdit training.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--dataset-name", default="ETTh1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--train-families", type=int, default=96)
    parser.add_argument("--valid-families", type=int, default=24)
    parser.add_argument("--test-families", type=int, default=24)
    args = parser.parse_args()

    result = build_family_dataset(
        csv_path=args.csv_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        random_seed=args.random_seed,
        train_families=args.train_families,
        valid_families=args.valid_families,
        test_families=args.test_families,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
