from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from test_scripts.prepare_forecast_revision_calibration_framework import (  # noqa: E402
    _load_config,
    _validate_config,
    build_plan,
)


def _check(path: str, status: str, message: str, category: str) -> dict:
    return {
        "path": path,
        "status": status,
        "message": message,
        "category": category,
    }


def _artifact_set(plan: dict) -> set[str]:
    artifacts = set()
    train_model = plan.get("artifacts", {}).get("train_model_path")
    heldout = plan.get("artifacts", {}).get("heldout_benchmark")
    if train_model:
        artifacts.add(train_model)
    if heldout:
        artifacts.add(heldout)
    for stage in plan.get("stages", []):
        for value in stage.get("artifacts", {}).values():
            if value:
                artifacts.add(value)
    return artifacts


def _load_plan(config_path: str | None, plan_path: str | None) -> tuple[dict, str]:
    if config_path:
        config = _load_config(config_path)
        _validate_config(config)
        return build_plan(config), config_path
    if not plan_path:
        raise ValueError("Either --config or --plan is required")
    return json.loads(Path(plan_path).read_text(encoding="utf-8")), plan_path


def run_preflight(config_path: str | None = None, plan_path: str | None = None) -> dict:
    plan, source = _load_plan(config_path, plan_path)
    known_artifacts = _artifact_set(plan)
    checks: list[dict] = []

    benchmark = plan.get("benchmark_path")
    if benchmark:
        bench_path = Path(benchmark)
        checks.append(
            _check(
                benchmark,
                "ok" if bench_path.exists() else "error",
                "base benchmark exists" if bench_path.exists() else "base benchmark missing",
                "benchmark",
            )
        )

    output_root = Path(plan["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    checks.append(_check(str(output_root), "ok", "output_root is writable", "output"))

    for stage in plan.get("stages", []):
        output_dir = Path(stage["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        checks.append(_check(str(output_dir), "ok", f"stage output dir ready: {stage['name']}", "output"))

        bench = stage.get("benchmark_path")
        if bench:
            bench_path = Path(bench)
            if bench_path.exists():
                checks.append(_check(bench, "ok", f"stage benchmark exists: {stage['name']}", "benchmark"))
            elif bench in known_artifacts:
                checks.append(_check(bench, "deferred", f"stage benchmark will be produced by an earlier stage: {stage['name']}", "dependency"))
            else:
                checks.append(_check(bench, "error", f"stage benchmark missing and not declared as produced artifact: {stage['name']}", "benchmark"))

        model = stage.get("calibration_model_path")
        if model:
            model_path = Path(model)
            if model_path.exists():
                checks.append(_check(model, "ok", f"calibration model exists: {stage['name']}", "model"))
            elif model in known_artifacts:
                checks.append(_check(model, "deferred", f"calibration model will be produced by an earlier stage: {stage['name']}", "dependency"))
            else:
                checks.append(_check(model, "error", f"calibration model missing and not declared as produced artifact: {stage['name']}", "model"))

        for artifact_name, artifact_path in stage.get("artifacts", {}).items():
            parent = Path(artifact_path).parent
            parent.mkdir(parents=True, exist_ok=True)
            checks.append(_check(str(parent), "ok", f"artifact parent ready for {artifact_name}: {stage['name']}", "output"))

    counts = {
        "ok": sum(item["status"] == "ok" for item in checks),
        "deferred": sum(item["status"] == "deferred" for item in checks),
        "warning": sum(item["status"] == "warning" for item in checks),
        "error": sum(item["status"] == "error" for item in checks),
    }
    return {
        "source": source,
        "experiment_id": plan.get("experiment_id"),
        "status": "pass" if counts["error"] == 0 else "fail",
        "counts": counts,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a preflight check for a forecast-revision calibration config or plan.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--plan", default=None)
    args = parser.parse_args()
    if not args.config and not args.plan:
        raise ValueError("Either --config or --plan must be provided")
    payload = run_preflight(config_path=args.config, plan_path=args.plan)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
