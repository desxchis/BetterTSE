from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from forecasting.data_utils import inspect_csv_frame, materialize_ltsf_compatible_csv

REPO_ROOT = Path(__file__).resolve().parent.parent
LTSF_LINEAR_ROOT = REPO_ROOT / "forecasting" / "baselines" / "vendor" / "LTSF-Linear"

TSLIB_MODEL_NAMES = {
    "dlinear_tslib": "DLinear",
    "patchtst_tslib": "PatchTST",
    "itransformer_tslib": "iTransformer",
    "timemixer_tslib": "TimeMixer",
    "autoformer_tslib": "Autoformer",
}

TSLIB_DATA_NAMES = {
    "traffic": "custom",
    "weather": "custom",
    "etth1": "ETTh1",
    "etth2": "ETTh2",
    "ettm1": "ETTm1",
    "ettm2": "ETTm2",
    "electricity": "custom",
}

DATA_FREQ = {
    "traffic": "H",
    "weather": "10min",
    "etth1": "H",
    "etth2": "H",
    "ettm1": "15min",
    "ettm2": "15min",
    "electricity": "H",
}

PANDAS_FREQ = {
    "traffic": "h",
    "weather": "10min",
    "etth1": "h",
    "etth2": "h",
    "ettm1": "15min",
    "ettm2": "15min",
    "electricity": "h",
}

MODEL_DEFAULTS = {
    "DLinear": {"features": "M"},
    "PatchTST": {"features": "M", "patch_len": 16, "stride": 8},
    "iTransformer": {"features": "M"},
    "TimeMixer": {"features": "M"},
    "Autoformer": {"features": "M", "label_len_ratio": 0.5},
}


def _tslib_model_name(baseline_name: str) -> str:
    if baseline_name not in TSLIB_MODEL_NAMES:
        raise ValueError(f"unsupported TSLib baseline: {baseline_name}")
    return TSLIB_MODEL_NAMES[baseline_name]


def _tslib_data_name(dataset_id: str | None) -> str:
    if not dataset_id:
        return "custom"
    return TSLIB_DATA_NAMES.get(str(dataset_id).lower(), "custom")


def _default_freq(dataset_id: str | None) -> str:
    if not dataset_id:
        return "h"
    return DATA_FREQ.get(str(dataset_id).lower(), "h")


def _default_pandas_freq(dataset_id: str | None) -> str:
    if not dataset_id:
        return "h"
    return PANDAS_FREQ.get(str(dataset_id).lower(), "h")


def build_tslib_training_manifest(
    *,
    baseline_name: str,
    dataset_id: str | None,
    csv_path: str,
    context_length: int,
    prediction_length: int,
    epochs: int,
    batch_size: int,
    lr: float,
    output_dir: str | Path,
    training_split_id: str,
    split_policy: str,
    feature: str,
    tslib_root: str | None = None,
) -> Dict[str, Any]:
    model_name = _tslib_model_name(baseline_name)
    frame = inspect_csv_frame(csv_path)
    num_features = int(frame["num_features"])
    dataset_key = str(dataset_id or Path(csv_path).stem).lower()
    data_name = _tslib_data_name(dataset_id)
    label_len = max(1, context_length // 2)
    model_id = f"{dataset_key}_{context_length}_{prediction_length}_{training_split_id}"
    checkpoint_dir = str(Path(output_dir).resolve())
    target = str(frame["default_target_col"] or feature)
    freq = _default_freq(dataset_id)
    backend = "tslib_template"
    cwd = str(REPO_ROOT)
    entrypoint = "run.py"

    if model_name in {"DLinear", "Autoformer"}:
        backend = "ltsf_linear_vendor"
        cwd = str(LTSF_LINEAR_ROOT.resolve())
        entrypoint = "run_longExp.py"
    elif tslib_root:
        cwd = str(Path(tslib_root).expanduser().resolve())
        entrypoint = str((Path(cwd) / "run.py").resolve())

    effective_csv_path = str(Path(csv_path).resolve())
    if data_name == "custom":
        prepared_dir = Path(output_dir).resolve() / "tslib_data"
        prepared_name = f"{dataset_key}_custom_ltsf.csv"
        prepared = materialize_ltsf_compatible_csv(
            csv_path,
            prepared_dir / prepared_name,
            pandas_freq=_default_pandas_freq(dataset_id),
        )
        effective_csv_path = prepared["csv_path"]
        target = str(prepared["target_col"])

    root_path = str(Path(effective_csv_path).resolve().parent)
    data_path = Path(effective_csv_path).name

    cmd = [
        "python",
        entrypoint,
        "--is_training",
        "1",
        "--model_id",
        model_id,
        "--model",
        model_name,
        "--data",
        data_name,
        "--root_path",
        root_path,
        "--data_path",
        data_path,
        "--features",
        str(MODEL_DEFAULTS[model_name].get("features", "M")),
        "--target",
        target,
        "--freq",
        freq,
        "--checkpoints",
        checkpoint_dir,
        "--seq_len",
        str(context_length),
        "--label_len",
        str(label_len),
        "--pred_len",
        str(prediction_length),
        "--enc_in",
        str(num_features),
        "--dec_in",
        str(num_features),
        "--c_out",
        str(num_features),
        "--batch_size",
        str(batch_size),
        "--learning_rate",
        str(lr),
        "--train_epochs",
        str(epochs),
        "--itr",
        "1",
        "--des",
        "BetterTSEPaper",
    ]
    if model_name == "PatchTST":
        cmd.extend([
            "--task_name",
            "long_term_forecast",
            "--patch_len",
            str(MODEL_DEFAULTS[model_name]["patch_len"]),
            "--stride",
            str(MODEL_DEFAULTS[model_name]["stride"]),
        ])

    notes = [
        "This manifest is generated by BetterTSE to keep paper baselines on a unified training contract.",
        "The baseline directory is materialized before training; attach exported inference artifacts after external training finishes.",
    ]
    if backend == "ltsf_linear_vendor":
        notes.append("This launcher is directly runnable against the vendored LTSF-Linear code already present in this repo.")
    else:
        notes.append("Traffic, Weather, and Electricity are mapped to custom csv mode in the generic TSLib command template.")

    return {
        "baseline_source": "tslib",
        "baseline_name": baseline_name,
        "tslib_model_name": model_name,
        "training_backend": backend,
        "dataset_id": dataset_id,
        "data_name": data_name,
        "csv_path": str(Path(csv_path).resolve()),
        "effective_csv_path": effective_csv_path,
        "root_path": root_path,
        "data_path": data_path,
        "target": target,
        "freq": freq,
        "num_features": num_features,
        "feature_columns": frame["feature_columns"],
        "feature": feature,
        "seq_len": context_length,
        "label_len": label_len,
        "pred_len": prediction_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "split_policy": split_policy,
        "training_split_id": training_split_id,
        "checkpoint_dir": checkpoint_dir,
        "cwd": cwd,
        "tslib_root": None if tslib_root is None else str(Path(tslib_root).expanduser().resolve()),
        "command": cmd,
        "command_shell": " ".join(cmd),
        "export_ready": False,
        "notes": notes,
    }


def write_tslib_training_manifest(output_dir: str | Path, manifest: Dict[str, Any]) -> Dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "tslib_training_manifest.json"
    sh_path = output / "launch_tslib_train.sh"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    shell_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by BetterTSE forecasting/tslib_bridge.py",
        f"cd {manifest['cwd']}",
        manifest["command_shell"],
        "",
    ]
    sh_path.write_text("\n".join(shell_lines), encoding="utf-8")
    try:
        sh_path.chmod(0o755)
    except OSError:
        pass
    return {
        "manifest_json": str(json_path),
        "launch_script": str(sh_path),
    }
