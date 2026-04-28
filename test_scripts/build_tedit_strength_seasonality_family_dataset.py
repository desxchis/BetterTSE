from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from test_scripts.build_tedit_strength_trend_family_dataset import build_family_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a seasonality discrete strength family dataset for TEdit smoke checks.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--dataset-name", default="ETTh1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--collection-root", default="")
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
        injection_types=["seasonality_injection"],
        selector="seasonality_injection",
        collection_root=args.collection_root.strip() or None,
    )

    output_dir = Path(result["output_dir"])
    readme_path = output_dir / "README.md"
    if readme_path.exists():
        lines = readme_path.read_text(encoding="utf-8").splitlines()
        if lines:
            lines[0] = "# Seasonality Strength Family Dataset"
            readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
