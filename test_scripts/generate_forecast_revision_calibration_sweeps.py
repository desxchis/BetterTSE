from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _slug(value: object) -> str:
    text = str(value)
    text = text.replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    return text.strip("_") or "value"


def _set_nested(payload: dict, dotted_key: str, value: object) -> None:
    parts = dotted_key.split(".")
    cursor = payload
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def generate_sweeps(base_config_path: str, sweep_spec: dict, output_dir: str) -> dict:
    base = _load_json(base_config_path)
    keys = list(sweep_spec.keys())
    values = [list(sweep_spec[key]) for key in keys]
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, combo in enumerate(itertools.product(*values), start=1):
        payload = json.loads(json.dumps(base))
        suffix_parts = []
        for key, value in zip(keys, combo):
            _set_nested(payload, key, value)
            suffix_parts.append(f"{key.split('.')[-1]}_{_slug(value)}")
        suffix = "__".join(suffix_parts)
        payload["experiment_id"] = f"{base['experiment_id']}__{suffix}"
        if payload.get("title"):
            payload["title"] = f"{payload['title']} [{suffix}]"
        if payload.get("output_root"):
            payload["output_root"] = f"{payload['output_root']}__{suffix}"
        payload["sweep_parent_experiment_id"] = base["experiment_id"]
        payload["sweep_variant_index"] = idx
        payload["sweep_overrides"] = dict(zip(keys, combo))

        config_path = out_root / f"{payload['experiment_id']}.json"
        config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows.append({
            "experiment_id": payload["experiment_id"],
            "config_path": str(config_path),
            "output_root": payload.get("output_root"),
            "overrides": payload["sweep_overrides"],
        })

    index = {
        "base_config": base_config_path,
        "output_dir": output_dir,
        "sweep_spec": sweep_spec,
        "variants": rows,
    }
    (out_root / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sweep variants for forecast-revision calibration configs.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sweep-file", default=None)
    parser.add_argument("--sweep-json", default=None)
    args = parser.parse_args()
    if not args.sweep_file and not args.sweep_json:
        raise ValueError("Either --sweep-file or --sweep-json must be provided")
    sweep_spec = _load_json(args.sweep_file) if args.sweep_file else json.loads(args.sweep_json)
    payload = generate_sweeps(args.base_config, sweep_spec, args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
