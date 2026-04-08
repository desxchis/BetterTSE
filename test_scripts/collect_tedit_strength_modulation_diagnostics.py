from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tool.tedit_wrapper import TEditWrapper


STRENGTHS = [("weak", 0), ("medium", 1), ("strong", 2)]


def _trend_attrs(direction: str) -> tuple[np.ndarray, np.ndarray]:
    src_attrs = np.array([0, 0, 0], dtype=np.int64)
    if direction == "down":
        tgt_attrs = np.array([1, 0, 1], dtype=np.int64)
    else:
        tgt_attrs = np.array([1, 1, 1], dtype=np.int64)
    return src_attrs, tgt_attrs


def _family_direction(family: Dict[str, Any]) -> str:
    for sample in family.get("samples", []):
        direction = str(sample.get("direction", ""))
        if direction in {"up", "down"}:
            return direction
        if direction in {"upward", "downward"}:
            return "up" if direction == "upward" else "down"
    return "up"


def _aggregate_layer_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    out = {}
    for key in rows[0].keys():
        values = [float(row[key]) for row in rows]
        out[key] = float(np.mean(values))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect layer-wise strength modulation diagnostics.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-families", type=int, default=12)
    parser.add_argument("--edit-steps", type=int, default=10)
    parser.add_argument("--sampler", default="ddim")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    families = [fam for fam in benchmark.get("families", []) if str(fam.get("tool_name")) == "trend_injection"][: args.max_families]
    if not families:
        raise ValueError("No trend_injection families found in benchmark.")

    wrapper = TEditWrapper(model_path=args.model_path, config_path=args.config_path, device=args.device)
    wrapper.set_edit_steps(args.edit_steps)
    model = wrapper.model
    output_path = Path(args.output)
    layer_strength_rows: Dict[str, Dict[int, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    family_payloads = []

    with torch.no_grad():
        for family_idx, family in enumerate(families):
            direction = _family_direction(family)
            src_attrs_np, tgt_attrs_np = _trend_attrs(direction)
            samples = {str(sample["strength_text"]): sample for sample in family.get("samples", [])}
            source_ts = np.asarray(samples["weak"]["source_ts"], dtype=np.float32)
            seq_len = int(source_ts.shape[0])
            src_x = torch.from_numpy(source_ts.reshape(1, seq_len, 1)).to(args.device)
            tp = torch.zeros(1, seq_len, device=args.device)
            src_attrs = torch.from_numpy(src_attrs_np).unsqueeze(0).to(args.device)
            tgt_attrs = torch.from_numpy(tgt_attrs_np).unsqueeze(0).to(args.device)
            side_emb = model.side_en(tp)
            src_attr_emb = model.attr_en(src_attrs)
            tgt_attr_emb = model.attr_en(tgt_attrs)
            family_record = {"family_id": family.get("family_id"), "per_strength": {}}

            for strength_name, strength_label in STRENGTHS:
                instruction_text = str(samples[strength_name]["instruction_text"])
                torch.manual_seed(args.seed + family_idx)
                np.random.seed(args.seed + family_idx)
                model._edit(
                    src_x=src_x.permute(0, 2, 1),
                    side_emb=side_emb,
                    src_attr_emb=src_attr_emb,
                    tgt_attr_emb=tgt_attr_emb,
                    sampler=args.sampler,
                    strength_label=torch.tensor([strength_label], device=args.device, dtype=torch.long),
                    task_id=None,
                    text_context=[instruction_text],
                    collect_strength_diagnostics=True,
                )
                trace = model.get_last_strength_trace() or []
                layer_rows = defaultdict(list)
                for step_info in trace:
                    for layer in step_info.get("layers", []):
                        layer_idx = int(layer["layer_idx"])
                        row = {k: float(v) for k, v in layer.items() if k != "layer_idx"}
                        layer_rows[layer_idx].append(row)
                        layer_strength_rows[strength_name][layer_idx].append(row)
                family_record["per_strength"][strength_name] = {
                    "num_steps": len(trace),
                    "layer_means": {
                        str(layer_idx): _aggregate_layer_rows(rows)
                        for layer_idx, rows in sorted(layer_rows.items())
                    },
                }
            family_payloads.append(family_record)

    summary = {"per_strength": {}, "pairwise_abs_gap": {}}
    strength_layer_means: Dict[str, Dict[int, Dict[str, float]]] = {}
    for strength_name in layer_strength_rows:
        strength_layer_means[strength_name] = {
            layer_idx: _aggregate_layer_rows(rows)
            for layer_idx, rows in sorted(layer_strength_rows[strength_name].items())
        }
        summary["per_strength"][strength_name] = {
            str(layer_idx): values for layer_idx, values in strength_layer_means[strength_name].items()
        }

    for lhs, rhs in [("weak", "medium"), ("medium", "strong"), ("weak", "strong")]:
        pair_key = f"{lhs}_vs_{rhs}"
        summary["pairwise_abs_gap"][pair_key] = {}
        lhs_layers = strength_layer_means.get(lhs, {})
        rhs_layers = strength_layer_means.get(rhs, {})
        for layer_idx in sorted(set(lhs_layers) & set(rhs_layers)):
            gap = {}
            for metric_name in lhs_layers[layer_idx]:
                gap[metric_name] = float(abs(lhs_layers[layer_idx][metric_name] - rhs_layers[layer_idx][metric_name]))
            summary["pairwise_abs_gap"][pair_key][str(layer_idx)] = gap

    payload = {
        "config": {
            "benchmark": args.benchmark,
            "model_path": args.model_path,
            "config_path": args.config_path,
            "max_families": args.max_families,
            "edit_steps": args.edit_steps,
            "sampler": args.sampler,
            "seed": args.seed,
            "device": args.device,
        },
        "summary": summary,
        "families": family_payloads,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
