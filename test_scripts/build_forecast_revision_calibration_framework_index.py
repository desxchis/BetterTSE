from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _plan_paths(output_root: str) -> tuple[Path, Path]:
    root = Path(output_root)
    return root / "experiment_plan.json", root / "README.md"


def build_index(config_dir: str, output_dir: str) -> dict:
    configs = sorted(Path(config_dir).glob("*.json"))
    rows = []
    for config_path in configs:
        config = _load_json(config_path)
        plan_json, plan_readme = _plan_paths(config["output_root"])
        stage_count = 0
        stage_names: list[str] = []
        if plan_json.exists():
            plan = _load_json(plan_json)
            stage_names = [stage.get("name", "") for stage in plan.get("stages", [])]
            stage_count = len(stage_names)
        rows.append({
            "priority": config.get("priority"),
            "experiment_id": config["experiment_id"],
            "title": config.get("title", config["experiment_id"]),
            "dataset_role": config.get("dataset_role", "unspecified"),
            "status": config.get("status", "planned"),
            "benchmark_path": config["benchmark_path"],
            "output_root": config["output_root"],
            "plan_json": str(plan_json) if plan_json.exists() else None,
            "plan_readme": str(plan_readme) if plan_readme.exists() else None,
            "stage_count": stage_count,
            "stage_names": stage_names,
        })
    rows.sort(key=lambda row: (row["priority"] is None, row["priority"], row["experiment_id"]))

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_dir": config_dir,
        "output_dir": output_dir,
        "rows": rows,
    }
    (out_root / "index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Calibration Framework Plan Index",
        "",
        "| Priority | Experiment | Role | Status | Planned Stages | Plan README |",
        "| ---: | --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {priority} | {title} | {dataset_role} | {status} | {stage_count} | {plan_link} |".format(
                priority=row["priority"] if row["priority"] is not None else "-",
                title=row["title"],
                dataset_role=row["dataset_role"],
                status=row["status"],
                stage_count=row["stage_count"],
                plan_link=row["plan_readme"] or "",
            )
        )
    lines.extend([
        "",
        "Recommended execution order is encoded in `priority`.",
        "",
    ])
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a top-level index for calibration framework plans.")
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    payload = build_index(args.config_dir, args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
