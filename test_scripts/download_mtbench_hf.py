from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from huggingface_hub import hf_hub_download, snapshot_download


DEFAULT_DATASET = "GGLabYale/MTBench_finance_aligned_pairs_short"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a local MTBench dataset artifact from Hugging Face.")
    parser.add_argument("--repo-id", default=DEFAULT_DATASET)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--filename", default=None, help="Single file to download, e.g. data/train-00000-of-00001.parquet")
    parser.add_argument("--all-files", action="store_true", help="Download the full dataset snapshot instead of one file")
    parser.add_argument("--local-dir-use-symlinks", action="store_true", help="Keep Hugging Face symlink behavior")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_files:
        local_path = snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            local_dir=str(output_dir),
            local_dir_use_symlinks=args.local_dir_use_symlinks,
        )
        payload = {
            "repo_id": args.repo_id,
            "mode": "snapshot",
            "local_path": local_path,
        }
    else:
        if not args.filename:
            raise SystemExit("--filename is required unless --all-files is set")
        local_path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            filename=args.filename,
            local_dir=str(output_dir),
            local_dir_use_symlinks=args.local_dir_use_symlinks,
        )
        payload = {
            "repo_id": args.repo_id,
            "mode": "single_file",
            "filename": args.filename,
            "local_path": local_path,
        }

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
