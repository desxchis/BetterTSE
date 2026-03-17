from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def _load_config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ensure_list(value: list[str] | None) -> list[str]:
    return list(value or [])


def _quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _resolve_benchmark(source: str, benchmark_path: str, heldout_benchmark: str | None) -> str:
    if source == "benchmark":
        return benchmark_path
    if source == "heldout":
        if not heldout_benchmark:
            raise ValueError("Config requested benchmark_source=heldout but no train stage is defined")
        return heldout_benchmark
    raise ValueError(f"Unsupported benchmark_source: {source}")


def _resolve_model(source: str | None, explicit_path: str | None, train_model_path: str | None) -> str | None:
    if source is None:
        return explicit_path
    if source == "train_model":
        if not train_model_path:
            raise ValueError("Config requested calibration_model_source=train_model but no train stage is defined")
        return train_model_path
    if source == "explicit":
        if not explicit_path:
            raise ValueError("Config requested calibration_model_source=explicit but no explicit path was provided")
        return explicit_path
    raise ValueError(f"Unsupported calibration_model_source: {source}")


def _train_stage(
    benchmark_path: str,
    output_root: Path,
    train_cfg: dict,
) -> tuple[dict, str, str]:
    output_dir = output_root / train_cfg.get("output_subdir", "train_learned_linear")
    command = [
        sys.executable,
        "test_scripts/train_forecast_revision_calibrator.py",
        "--benchmark",
        benchmark_path,
        "--output-dir",
        str(output_dir),
        "--train-ratio",
        str(train_cfg.get("train_ratio", 0.8)),
        "--seed",
        str(train_cfg.get("seed", 7)),
        "--alpha",
        str(train_cfg.get("alpha", 1.0)),
    ]
    model_path = str(output_dir / "learned_linear_calibrator.json")
    heldout_path = str(output_dir / "heldout_benchmark.json")
    stage = {
        "name": train_cfg.get("name", "train_learned_linear"),
        "type": "train",
        "output_dir": str(output_dir),
        "artifacts": {
            "model_path": model_path,
            "heldout_benchmark": heldout_path,
        },
        "command": command,
    }
    return stage, model_path, heldout_path


def _oracle_stage(
    benchmark_path: str,
    output_root: Path,
    stage_cfg: dict,
    train_model_path: str | None,
    heldout_benchmark: str | None,
) -> dict:
    output_dir = output_root / stage_cfg.get("output_subdir", stage_cfg.get("name", "oracle"))
    resolved_benchmark = _resolve_benchmark(
        stage_cfg.get("benchmark_source", "benchmark"),
        benchmark_path,
        heldout_benchmark,
    )
    calibration_model = _resolve_model(
        stage_cfg.get("calibration_model_source"),
        stage_cfg.get("calibration_model"),
        train_model_path,
    )
    command = [
        sys.executable,
        "test_scripts/run_forecast_revision_calibration_benchmark.py",
        "--benchmark",
        resolved_benchmark,
        "--output-dir",
        str(output_dir),
    ]
    max_samples = stage_cfg.get("max_samples")
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])
    methods = _ensure_list(stage_cfg.get("methods"))
    if methods:
        command.append("--methods")
        command.extend(methods)
    if calibration_model:
        command.extend(["--calibration-model", calibration_model])
    return {
        "name": stage_cfg.get("name", "oracle"),
        "type": "oracle_benchmark",
        "output_dir": str(output_dir),
        "benchmark_path": resolved_benchmark,
        "calibration_model_path": calibration_model,
        "command": command,
    }


def _semi_oracle_stage(
    benchmark_path: str,
    output_root: Path,
    stage_cfg: dict,
    train_model_path: str | None,
    heldout_benchmark: str | None,
) -> dict:
    output_dir = output_root / stage_cfg.get("output_subdir", stage_cfg.get("name", "semi_oracle"))
    resolved_benchmark = _resolve_benchmark(
        stage_cfg.get("benchmark_source", "benchmark"),
        benchmark_path,
        heldout_benchmark,
    )
    calibration_model = _resolve_model(
        stage_cfg.get("calibration_model_source"),
        stage_cfg.get("calibration_model"),
        train_model_path,
    )
    command = [
        sys.executable,
        "test_scripts/run_forecast_revision_semi_oracle_suite.py",
        "--benchmark",
        resolved_benchmark,
        "--output-dir",
        str(output_dir),
        "--calibration-strategy",
        stage_cfg.get("calibration_strategy", "rule_local_stats"),
    ]
    max_samples = stage_cfg.get("max_samples")
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])
    if calibration_model:
        command.extend(["--calibration-model", calibration_model])
    return {
        "name": stage_cfg.get("name", "semi_oracle"),
        "type": "semi_oracle_suite",
        "output_dir": str(output_dir),
        "benchmark_path": resolved_benchmark,
        "calibration_strategy": stage_cfg.get("calibration_strategy", "rule_local_stats"),
        "calibration_model_path": calibration_model,
        "command": command,
    }


def build_plan(config: dict, output_root_override: str | None = None) -> dict:
    benchmark_path = config["benchmark_path"]
    output_root = Path(output_root_override or config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    stages: list[dict] = []
    train_model_path: str | None = None
    heldout_benchmark: str | None = None

    train_cfg = config.get("train_stage")
    if train_cfg and train_cfg.get("enabled", True):
        stage, train_model_path, heldout_benchmark = _train_stage(
            benchmark_path=benchmark_path,
            output_root=output_root,
            train_cfg=train_cfg,
        )
        stages.append(stage)

    oracle_cfg = config.get("oracle_stage")
    if oracle_cfg and oracle_cfg.get("enabled", True):
        stages.append(
            _oracle_stage(
                benchmark_path=benchmark_path,
                output_root=output_root,
                stage_cfg=oracle_cfg,
                train_model_path=train_model_path,
                heldout_benchmark=heldout_benchmark,
            )
        )

    for stage_cfg in config.get("semi_oracle_stages", []):
        if not stage_cfg.get("enabled", True):
            continue
        stages.append(
            _semi_oracle_stage(
                benchmark_path=benchmark_path,
                output_root=output_root,
                stage_cfg=stage_cfg,
                train_model_path=train_model_path,
                heldout_benchmark=heldout_benchmark,
            )
        )

    return {
        "experiment_id": config["experiment_id"],
        "benchmark_path": benchmark_path,
        "output_root": str(output_root),
        "artifacts": {
            "train_model_path": train_model_path,
            "heldout_benchmark": heldout_benchmark,
        },
        "stages": stages,
    }


def _render_readme(plan: dict) -> str:
    lines = [
        "# Forecast Revision Calibration Framework Plan",
        "",
        f"- experiment_id: `{plan['experiment_id']}`",
        f"- benchmark_path: `{plan['benchmark_path']}`",
        f"- output_root: `{plan['output_root']}`",
        "",
        "## Planned Stages",
        "",
    ]
    for idx, stage in enumerate(plan["stages"], start=1):
        lines.append(f"{idx}. `{stage['name']}`")
        lines.append(f"   - type: `{stage['type']}`")
        lines.append(f"   - output_dir: `{stage['output_dir']}`")
        if stage.get("benchmark_path"):
            lines.append(f"   - benchmark: `{stage['benchmark_path']}`")
        if stage.get("calibration_strategy"):
            lines.append(f"   - calibration_strategy: `{stage['calibration_strategy']}`")
        if stage.get("calibration_model_path"):
            lines.append(f"   - calibration_model: `{stage['calibration_model_path']}`")
        lines.append(f"   - command: `{_quote_command(stage['command'])}`")
    lines.extend([
        "",
        "## Artifacts",
        "",
        f"- train_model_path: `{plan['artifacts'].get('train_model_path')}`",
        f"- heldout_benchmark: `{plan['artifacts'].get('heldout_benchmark')}`",
        "",
        "This file is generated by `test_scripts/prepare_forecast_revision_calibration_framework.py`.",
        "",
    ])
    return "\n".join(lines)


def _render_shell(plan: dict) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    for stage in plan["stages"]:
        lines.append(f"# {stage['name']}")
        lines.append(_quote_command(stage["command"]))
        lines.append("")
    return "\n".join(lines)


def write_plan_artifacts(plan: dict) -> dict:
    output_root = Path(plan["output_root"])
    manifest_path = output_root / "experiment_plan.json"
    manifest_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    shell_path = output_root / "run_commands.sh"
    shell_path.write_text(_render_shell(plan), encoding="utf-8")
    readme_path = output_root / "README.md"
    readme_path.write_text(_render_readme(plan), encoding="utf-8")
    return {
        "manifest_path": str(manifest_path),
        "shell_path": str(shell_path),
        "readme_path": str(readme_path),
    }


def execute_plan(plan: dict) -> None:
    for stage in plan["stages"]:
        subprocess.run(stage["command"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a config-driven forecast-revision calibration experiment plan.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    plan = build_plan(config, output_root_override=args.output_root)
    artifacts = write_plan_artifacts(plan)
    if args.execute:
        execute_plan(plan)

    print(json.dumps({
        "config": args.config,
        "execute": args.execute,
        "plan": plan,
        "artifacts": artifacts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
