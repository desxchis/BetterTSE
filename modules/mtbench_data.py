from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


MTBENCH_REQUIRED_COLUMNS = (
    "input_timestamps",
    "input_window",
    "output_timestamps",
    "output_window",
    "text",
    "trend",
    "technical",
    "alignment",
)


@dataclass
class MTBenchAlignedPair:
    input_timestamps: List[float]
    input_window: List[float]
    output_timestamps: List[float]
    output_window: List[float]
    text: Dict[str, Any]
    trend: Dict[str, Any]
    technical: Dict[str, Any]
    alignment: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _maybe_parse_json_like(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    if text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return value
    return value


def _as_float_list(value: Any) -> List[float]:
    parsed = _maybe_parse_json_like(value)
    if parsed is None:
        return []
    if isinstance(parsed, np.ndarray):
        parsed = parsed.tolist()
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Expected list-like value, got {type(parsed).__name__}")
    result: List[float] = []
    for item in parsed:
        result.append(float(item))
    return result


def _as_dict(value: Any) -> Dict[str, Any]:
    parsed = _maybe_parse_json_like(value)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected dict-like value, got {type(parsed).__name__}")
    return parsed


def _read_table(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        return pd.read_json(file_path)
    if suffix == ".jsonl":
        return pd.read_json(file_path, lines=True)
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".parquet":
        try:
            return pd.read_parquet(file_path)
        except Exception as exc:
            raise RuntimeError(
                "Reading parquet requires a parquet engine such as pyarrow or fastparquet. "
                "Install one first or export MTBench to JSON/CSV."
            ) from exc
    raise ValueError(f"Unsupported MTBench file format: {suffix}")


def load_mtbench_aligned_pairs(path: str | Path, limit: int | None = None) -> List[MTBenchAlignedPair]:
    frame = _read_table(path)
    missing = [column for column in MTBENCH_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing MTBench columns: {missing}")

    if limit is not None:
        frame = frame.head(limit)

    records: List[MTBenchAlignedPair] = []
    for row in frame.to_dict(orient="records"):
        records.append(
            MTBenchAlignedPair(
                input_timestamps=_as_float_list(row["input_timestamps"]),
                input_window=_as_float_list(row["input_window"]),
                output_timestamps=_as_float_list(row["output_timestamps"]),
                output_window=_as_float_list(row["output_window"]),
                text=_as_dict(row["text"]),
                trend=_as_dict(row["trend"]),
                technical=_as_dict(row["technical"]),
                alignment=str(row.get("alignment", "")),
            )
        )
    return records


def summarize_mtbench_pairs(records: Iterable[MTBenchAlignedPair]) -> Dict[str, Any]:
    rows = list(records)
    if not rows:
        return {
            "count": 0,
            "input_length_min": 0,
            "input_length_max": 0,
            "output_length_min": 0,
            "output_length_max": 0,
            "alignment_values": [],
        }

    input_lengths = [len(row.input_window) for row in rows]
    output_lengths = [len(row.output_window) for row in rows]
    alignments = sorted({row.alignment for row in rows})
    text_keys = sorted({key for row in rows for key in row.text.keys()})
    trend_keys = sorted({key for row in rows for key in row.trend.keys()})
    technical_keys = sorted({key for row in rows for key in row.technical.keys()})

    return {
        "count": len(rows),
        "input_length_min": int(min(input_lengths)),
        "input_length_max": int(max(input_lengths)),
        "output_length_min": int(min(output_lengths)),
        "output_length_max": int(max(output_lengths)),
        "alignment_values": alignments,
        "text_keys": text_keys,
        "trend_keys": trend_keys,
        "technical_keys": technical_keys,
    }


def pair_to_revision_inputs(pair: MTBenchAlignedPair) -> Dict[str, Any]:
    return {
        "history_ts": list(pair.input_window),
        "future_gt": list(pair.output_window),
        "context_text": json.dumps(pair.text, ensure_ascii=False),
        "metadata": {
            "input_timestamps": pair.input_timestamps,
            "output_timestamps": pair.output_timestamps,
            "alignment": pair.alignment,
            "trend": pair.trend,
            "technical": pair.technical,
        },
    }
