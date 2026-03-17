from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from forecasting.registry import load_baseline


def _load_history_from_json(input_json: str) -> tuple[np.ndarray, int]:
    with open(input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    history = np.asarray(payload["history_ts"], dtype=np.float64)
    horizon = int(payload["forecast_horizon"])
    return history, horizon


def infer_baseline(model_name: str, model_dir: str, input_json: str, output_json: str) -> dict:
    history, horizon = _load_history_from_json(input_json)
    baseline = load_baseline(model_name, model_dir)
    pred = baseline.predict(history, horizon)
    result = {"model_name": model_name, "forecast_horizon": horizon, "prediction": pred.astype(float).tolist()}
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forecasting baseline inference.")
    parser.add_argument("--model-name", required=True, choices=["naive_last", "dlinear_like", "patchtst"])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    result = infer_baseline(
        model_name=args.model_name,
        model_dir=args.model_dir,
        input_json=args.input_json,
        output_json=args.output_json,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
