from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.mtbench_data import load_mtbench_aligned_pairs, pair_to_revision_inputs, summarize_mtbench_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a local MTBench aligned-pairs export.")
    parser.add_argument("--path", required=True, help="Path to MTBench export file (.json/.jsonl/.csv/.parquet)")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    records = load_mtbench_aligned_pairs(args.path, limit=args.limit)
    summary = summarize_mtbench_pairs(records)
    preview = [pair_to_revision_inputs(record) for record in records[: min(3, len(records))]]

    payload = {
        "path": args.path,
        "summary": summary,
        "preview": preview,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
