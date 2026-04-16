from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _best_row(rows: list[dict], key: str) -> dict | None:
    valid = [row for row in rows if row.get(key) is not None]
    if not valid:
        return None
    return max(valid, key=lambda row: row.get(key, float("-inf")))


def _aggregate_plan(plan_path: Path) -> dict:
    plan = _load_json(plan_path)
    stages = []
    complete_count = 0
    for stage in plan.get("stages", []):
        stage_result = {
            "name": stage["name"],
            "type": stage["type"],
            "output_dir": stage["output_dir"],
            "status": "missing",
        }
        out_dir = Path(stage["output_dir"])
        if stage["type"] == "train":
            artifacts = stage.get("artifacts", {})
            model_exists = Path(artifacts.get("model_path", "")).exists() if artifacts.get("model_path") else False
            heldout_exists = Path(artifacts.get("heldout_benchmark", "")).exists() if artifacts.get("heldout_benchmark") else False
            stage_result.update({
                "status": "complete" if model_exists and heldout_exists else "partial",
                "model_exists": model_exists,
                "heldout_exists": heldout_exists,
            })
        elif stage["type"] == "oracle_benchmark":
            summary_path = out_dir / "calibration_benchmark_summary.json"
            if summary_path.exists():
                summary = _load_json(summary_path)
                best = _best_row(summary.get("table_rows", []), "avg_revision_gain")
                stage_result.update({
                    "status": "complete",
                    "summary_path": str(summary_path),
                    "best_method": best.get("method") if best else None,
                    "best_revision_gain": best.get("avg_revision_gain") if best else None,
                })
            else:
                stage_result.update({"status": "missing", "summary_path": str(summary_path)})
        elif stage["type"] == "semi_oracle_suite":
            summary_path = out_dir / "semi_oracle_summary.json"
            if summary_path.exists():
                summary = _load_json(summary_path)
                best = _best_row(summary.get("table_rows", []), "avg_revision_gain")
                stage_result.update({
                    "status": "complete",
                    "summary_path": str(summary_path),
                    "best_mode": best.get("mode") if best else None,
                    "best_revision_gain": best.get("avg_revision_gain") if best else None,
                })
            else:
                stage_result.update({"status": "missing", "summary_path": str(summary_path)})
        if stage_result["status"] == "complete":
            complete_count += 1
        stages.append(stage_result)

    return {
        "experiment_id": plan.get("experiment_id"),
        "title": plan.get("title", plan.get("experiment_id")),
        "priority": plan.get("priority"),
        "dataset_role": plan.get("dataset_role"),
        "plan_path": str(plan_path),
        "complete_stage_count": complete_count,
        "stage_count": len(stages),
        "stages": stages,
    }


def aggregate(plan_index_path: str, output_dir: str | None = None) -> dict:
    index = _load_json(Path(plan_index_path))
    rows = []
    for row in index.get("rows", []):
        plan_json = row.get("plan_json")
        if not plan_json or not Path(plan_json).exists():
            continue
        rows.append(_aggregate_plan(Path(plan_json)))
    rows.sort(key=lambda item: (item.get("priority") is None, item.get("priority"), item.get("experiment_id")))
    payload = {
        "plan_index": plan_index_path,
        "experiments": rows,
    }
    if output_dir:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "aggregate_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [
            "# Calibration Result Aggregate",
            "",
            "| Priority | Experiment | Complete Stages | Total Stages | Top Completed Result |",
            "| ---: | --- | ---: | ---: | --- |",
        ]
        for item in rows:
            top = ""
            for stage in item.get("stages", []):
                if stage.get("status") == "complete":
                    if stage.get("best_method"):
                        top = f"{stage['name']}: {stage['best_method']} ({stage.get('best_revision_gain')})"
                    elif stage.get("best_mode"):
                        top = f"{stage['name']}: {stage['best_mode']} ({stage.get('best_revision_gain')})"
            lines.append(
                f"| {item.get('priority')} | {item.get('title')} | {item.get('complete_stage_count')} | {item.get('stage_count')} | {top} |"
            )
        (out_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate forecast-revision calibration results across framework plans.")
    parser.add_argument("--plan-index", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    payload = aggregate(args.plan_index, args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
