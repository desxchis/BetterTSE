from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    {
        "name": "current_ranking",
        "trainable_scope": "strength_only",
        "edit_region_loss_weight": 1.0,
        "background_loss_weight": 0.2,
        "monotonic_loss_weight": 0.5,
        "late_output_coupling_enabled": 0,
        "late_output_coupling_gain": 1.0,
    },
    {
        "name": "current_ranking_wider_unfreeze",
        "trainable_scope": "wider_unfreeze",
        "edit_region_loss_weight": 1.0,
        "background_loss_weight": 0.2,
        "monotonic_loss_weight": 0.5,
        "late_output_coupling_enabled": 0,
        "late_output_coupling_gain": 1.0,
    },
    {
        "name": "current_ranking_wider_unfreeze_late_output_coupling",
        "trainable_scope": "wider_unfreeze",
        "edit_region_loss_weight": 1.0,
        "background_loss_weight": 0.2,
        "monotonic_loss_weight": 0.5,
        "late_output_coupling_enabled": 1,
        "late_output_coupling_gain": 1.0,
    },
]


def _run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def _resolve_path(repo_root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    repo_candidate = (repo_root / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate
    main_candidate = (repo_root.parent / "BetterTSE-main" / candidate).resolve()
    if main_candidate.exists():
        return main_candidate
    return repo_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 4-way strength dynamics diagnostic matrix.")
    parser.add_argument("--pretrained-dir", required=True)
    parser.add_argument("--model-config-path", default="trend_types/model_configs.yaml")
    parser.add_argument("--pretrained-model-path", default="trend_types/ckpts/model_best.pth")
    parser.add_argument("--finetune-config-path", default="TEdit-main/configs/synthetic/finetune_strength_trend_family.yaml")
    parser.add_argument("--evaluate-config-path", default="tmp/strength_phase1/evaluate.synthetic.phase1b.yaml")
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-families", type=int, default=12)
    parser.add_argument("--edit-steps", type=int, default=10)
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--pretrained-run", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    tedit_root = repo_root / "TEdit-main"
    finetune_config_path = _resolve_path(repo_root, args.finetune_config_path)
    evaluate_config_path = _resolve_path(repo_root, args.evaluate_config_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    selected = set(args.experiment)
    experiments = [exp for exp in EXPERIMENTS if not selected or exp["name"] in selected]
    for experiment in experiments:
        exp_dir = output_dir / experiment["name"]
        exp_dir.mkdir(parents=True, exist_ok=True)
        train_cmd = [
            sys.executable,
            "run_finetune.py",
            "--pretrained_dir", str(Path(args.pretrained_dir)),
            "--model_config_path", args.model_config_path,
            "--pretrained_model_path", args.pretrained_model_path,
            "--finetune_config_path", str(finetune_config_path),
            "--evaluate_config_path", str(evaluate_config_path),
            "--data_type", "discrete_strength_family",
            "--data_folder", str(Path(args.data_folder)),
            "--save_folder", str(exp_dir),
            "--n_runs", str(args.n_runs),
            "--epochs", str(args.epochs),
            "--lr", "1e-5",
            "--strength-lr-scale", "5.0",
            "--strength-diagnostics", "1",
            "--strength-diagnostics-interval", "1",
            "--include_self", "0",
            "--trainable-scope", experiment["trainable_scope"],
            "--edit-region-loss-weight", str(experiment["edit_region_loss_weight"]),
            "--background-loss-weight", str(experiment["background_loss_weight"]),
            "--monotonic-loss-weight", str(experiment["monotonic_loss_weight"]),
            "--late-output-coupling-enabled", str(experiment["late_output_coupling_enabled"]),
            "--late-output-coupling-gain", str(experiment["late_output_coupling_gain"]),
            "--seed-list", "1,7" if args.n_runs == 2 else "1",
            "--pretrained-run", str(args.pretrained_run),
        ]
        _run(train_cmd, cwd=tedit_root)

        run_summaries = []
        for run_idx in range(args.n_runs):
            model_path = exp_dir / str(run_idx) / "trend_injection" / "ckpts" / "model_best.pth"
            config_path = exp_dir / str(run_idx) / "trend_injection" / "model_configs.yaml"
            eval_output = exp_dir / f"heldout_monotonic_eval_run{run_idx}.json"
            eval_cmd = [
                sys.executable,
                str(repo_root / "test_scripts" / "run_tedit_trend_monotonic_eval.py"),
                "--benchmark", str(Path(args.benchmark)),
                "--model-path", str(model_path),
                "--config-path", str(config_path),
                "--output", str(eval_output),
                "--max-families", str(args.max_families),
                "--edit-steps", str(args.edit_steps),
                "--device", args.device,
            ]
            _run(eval_cmd, cwd=repo_root)

            modulation_output = exp_dir / f"modulation_diagnostics_run{run_idx}.json"
            diag_cmd = [
                sys.executable,
                str(repo_root / "test_scripts" / "collect_tedit_strength_modulation_diagnostics.py"),
                "--benchmark", str(Path(args.benchmark)),
                "--model-path", str(model_path),
                "--config-path", str(config_path),
                "--output", str(modulation_output),
                "--max-families", str(args.max_families),
                "--edit-steps", str(args.edit_steps),
                "--device", args.device,
            ]
            _run(diag_cmd, cwd=repo_root)

            eval_payload = json.loads(eval_output.read_text(encoding="utf-8"))
            run_summaries.append(eval_payload["summary"])
        summary.append(
            {
                "name": experiment["name"],
                "trainable_scope": experiment["trainable_scope"],
                "monotonic_loss_weight": experiment["monotonic_loss_weight"],
                "late_output_coupling_enabled": experiment["late_output_coupling_enabled"],
                "late_output_coupling_gain": experiment["late_output_coupling_gain"],
                "strong_minus_weak_edit_gain_mean": [row["strong_minus_weak_edit_gain_mean"] for row in run_summaries],
                "monotonic_hit_rate": [row["monotonic_hit_rate"] for row in run_summaries],
                "preservation_pass_rate": [row["preservation_pass_rate"] for row in run_summaries],
            }
        )

    summary_path = output_dir / "diagnostic_matrix_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    aggregate_cmd = [
        sys.executable,
        str(repo_root / "test_scripts" / "summarize_strength_dynamics_matrix.py"),
        "--matrix-dir", str(output_dir),
        "--output-json", str(summary_path),
        "--output-md", str(output_dir / "diagnostic_matrix_summary.md"),
    ]
    _run(aggregate_cmd, cwd=repo_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
