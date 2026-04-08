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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_folder)
    ts = np.load(dataset_root / f"{args.split}_ts.npy")
    attrs = np.load(dataset_root / f"{args.split}_attrs_idx.npy")
    idx = _select_sample(attrs, args.sample_idx)

    base = ts[idx, 0].astype(np.float32)
    src_attrs = attrs[idx, 0].astype(np.int64)
    tgt_attrs = attrs[idx, 1].astype(np.int64)
    instruction_text = None
    if bool(args.use_dataset_instruction):
        instruction_text = _resolve_instruction_text(dataset_root, args.split, idx)

    wrapper = TEditWrapper(model_path=args.model_path, config_path=args.config_path, device=args.device)

    rows = []
    outputs = []
    for strength_label in [0, 1, 2]:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        edited = wrapper.edit_time_series(
            ts=base,
            src_attrs=src_attrs,
            tgt_attrs=tgt_attrs,
            n_samples=1,
            sampler="ddim",
            edit_steps=args.edit_steps,
            strength_label=strength_label,
            task_id=None if args.task_id < 0 else args.task_id,
            instruction_text=instruction_text,
        )[0]
        outputs.append(edited)
        rows.append(
            {
                "strength_label": int(strength_label),
                "peak_abs_delta": float(np.max(np.abs(edited - base))),
                "mean_abs_delta": float(np.mean(np.abs(edited - base))),
                "sum_delta": float(np.sum(edited - base)),
            }
        )

    payload = {
        "dataset_folder": str(dataset_root),
        "split": args.split,
        "probe_idx": int(idx),
        "src_attrs": src_attrs.tolist(),
        "tgt_attrs": tgt_attrs.tolist(),
        "instruction_text": instruction_text,
        "seed": int(args.seed),
        "task_id": None if args.task_id < 0 else int(args.task_id),
        "rows": rows,
        "diff_0_1_linf": float(np.max(np.abs(outputs[0] - outputs[1]))),
        "diff_1_2_linf": float(np.max(np.abs(outputs[1] - outputs[2]))),
        "diff_0_2_linf": float(np.max(np.abs(outputs[0] - outputs[2]))),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
