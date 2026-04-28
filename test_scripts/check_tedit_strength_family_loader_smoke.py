from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

import torch


ROOT = Path(__file__).resolve().parent.parent
DATASET_MODULE_PATH = ROOT / "TEdit-main" / "data" / "discrete_strength_family.py"


def _load_dataset_module():
    spec = importlib.util.spec_from_file_location("discrete_strength_family", DATASET_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load dataset module from {DATASET_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_loader_smoke(dataset_root: Path, split: str, batch_size: int) -> Dict[str, Any]:
    module = _load_dataset_module()
    dataset = module.DiscreteStrengthFamilyDataset(str(dataset_root))
    loader = dataset.get_loader(split=split, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    src_attrs = batch["src_attrs"]
    tgt_attrs = batch["tgt_attrs"]
    strength_label = batch["strength_label"]
    strength_scalar = batch["strength_scalar"]
    family_sizes = batch["family_sizes"]

    report = {
        "dataset_root": str(dataset_root),
        "split": split,
        "src_x_shape": list(batch["src_x"].shape),
        "tgt_x_shape": list(batch["tgt_x"].shape),
        "mask_gt_shape": list(batch["mask_gt"].shape),
        "src_attrs_shape": list(src_attrs.shape),
        "tgt_attrs_shape": list(tgt_attrs.shape),
        "family_sizes": family_sizes.tolist(),
        "family_valid": bool(batch["family_valid"]),
        "family_order_valid": bool(batch["family_order_valid"]),
        "strength_label": strength_label.tolist(),
        "strength_scalar": [float(v) for v in strength_scalar.tolist()],
        "seasonality_attr_increase": bool(torch.all(tgt_attrs[:, 2] > src_attrs[:, 2]).item()),
        "strength_label_triplet_ok": strength_label.tolist()[:3] == [0, 1, 2],
        "strength_scalar_triplet_ok": [round(float(v), 4) for v in strength_scalar.tolist()[:3]] == [0.0, 0.5, 1.0],
    }
    report["health_pass"] = bool(
        report["family_valid"]
        and report["family_order_valid"]
        and report["seasonality_attr_increase"]
        and report["strength_label_triplet_ok"]
        and report["strength_scalar_triplet_ok"]
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-check DiscreteStrengthFamilyDataset loader output.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = run_loader_smoke(Path(args.dataset_root), split=args.split, batch_size=args.batch_size)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
