import types
import unittest
import json
import tempfile
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
TEDIT_ROOT = REPO_ROOT / "TEdit-main"
if str(TEDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(TEDIT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.conditional_generator import ConditionalGenerator
from models.diffusion.diff_csdi_multipatch_weaver import Diff_CSDI_MultiPatch_Weaver_Parallel, ResidualBlock
from data import EditDataset
from train.finetuner import Finetuner
import test_scripts.evaluate_tedit_strength_effect as strength_effect_eval
import test_scripts.run_tedit_trend_monotonic_eval as trend_monotonic_eval
from test_scripts.build_strength_pipeline_replay_benchmark import build_replay_benchmark
from test_scripts.verify_strength_pipeline_summary import verify_summary
from test_scripts.build_tedit_strength_trend_family_dataset import build_family_dataset
from tool.tedit_wrapper import TEditWrapper
from tool.ts_editors import (
    _build_fixed_period_seasonality_scaffold,
    _build_hard_zero_scaffold,
    _build_multiplier_scaffold,
    _build_noise_injection_scaffold,
    _build_seasonality_amplitude_anchor,
    _build_seasonality_scaffold,
    _build_step_change_scaffold,
    _build_trend_injection_scaffold,
    _resolve_season_enhance_tgt_attrs,
    _resolve_seasonality_amplitude_gain,
)
from modules.llm import _apply_explicit_prompt_hints
from run_pipeline import (
    _apply_family_region_override,
    _apply_pipeline_mode,
    _benchmark_region,
    _build_condensed_execution_instruction,
    _direct_tool_choice,
    _family_consistent_localization_enabled,
    _family_region_source,
    _resolve_execution_instruction,
    _sample_replay_plan,
)


class TestOutputBranchCarrierContract(unittest.TestCase):
    def test_explicit_strength_prompt_hints_override_wrong_medium_intent(self):
        plan = {
            "intent": {"effect_family": "trend", "shape": "hump", "direction": "up", "strength": "medium"},
            "execution": {"tool_name": "hybrid_up", "canonical_tool": "trend_linear_up", "parameters": {"strength_label": 1, "strength_scalar": 1.0}},
            "parameters": {"strength_label": 1, "strength_scalar": 1.0},
        }
        normalized = _apply_explicit_prompt_hints(
            plan,
            "Edit only one short window of the target series. Make that segment show a temporary upward hump that returns toward baseline. Use strong strength and keep the non-edit region unchanged.",
            ts_length=96,
        )

        self.assertEqual(normalized["intent"]["strength"], "strong")
        self.assertEqual(normalized["parameters"]["strength_label"], 2)
        self.assertEqual(normalized["parameters"]["strength_scalar"], 2.0)

    def test_family_consistent_localization_helpers_enable_and_override_region(self):
        sample = {"family_id": "family_001", "pipeline_options": {"family_consistent_localization": True}}
        self.assertTrue(_family_consistent_localization_enabled(sample))
        self.assertEqual(_family_region_source(sample), "llm_first_sample")

        plan = {
            "parameters": {"region": [10, 30]},
            "localization": {"region": [10, 30]},
            "execution": {"parameters": {"region": [10, 30]}},
        }
        overridden = _apply_family_region_override(plan, [20, 40], "family_001", "family_cache")
        self.assertEqual(overridden["parameters"]["region"], [20, 40])
        self.assertEqual(overridden["localization"]["region"], [20, 40])
        self.assertEqual(overridden["execution"]["parameters"]["region"], [20, 40])
        self.assertEqual(overridden["pipeline_localization_policy"]["region_source"], "family_cache")
        self.assertEqual(overridden["pipeline_localization_policy"]["family_id"], "family_001")

    def test_benchmark_family_region_source_uses_gt_region(self):
        sample = {
            "family_id": "seasonality_family_001",
            "gt_start": 12,
            "gt_end": 34,
            "pipeline_options": {
                "family_consistent_localization": True,
                "family_region_source": "benchmark",
            },
        }
        self.assertEqual(_family_region_source(sample), "benchmark")
        self.assertEqual(_benchmark_region(sample, {}, 96), [12, 34])

    def test_build_condensed_execution_instruction_uses_canonical_short_phrase(self):
        trend_plan = {"intent": {"effect_family": "trend", "direction": "up", "strength": "strong", "duration": "medium"}}
        season_plan = {"intent": {"effect_family": "seasonality", "shape": "periodic", "strength": "weak", "duration": "medium"}}
        self.assertEqual(_build_condensed_execution_instruction(trend_plan), "Apply a strong upward trend edit in this medium window")
        self.assertEqual(_build_condensed_execution_instruction(season_plan), "Apply a weak seasonality enhancement in this medium window")

    def test_direct_edit_periodic_prompt_routes_to_season_enhance(self):
        tool_name, canonical_tool = _direct_tool_choice("Make the periodic peaks and troughs in this segment clearer")
        self.assertEqual(tool_name, "season_enhance")
        self.assertEqual(canonical_tool, "seasonality_enhance")

    def test_resolve_execution_instruction_prefers_planner_execution_phrase(self):
        plan = {
            "intent": {"effect_family": "trend", "direction": "up", "strength": "strong", "duration": "medium"},
            "execution": {"execution_phrase": "Apply a strong upward trend edit in this medium window"},
            "parameters": {"instruction_text": "old local fallback phrase"},
        }
        self.assertEqual(_resolve_execution_instruction(plan), "Apply a strong upward trend edit in this medium window")

    def test_replay_plan_mode_uses_benchmark_plan_without_llm_mutation(self):
        sample = {
            "replay_plan": {
                "tool_name": "season_enhance",
                "intent": {"effect_family": "seasonality", "shape": "periodic"},
                "parameters": {"region": [4, 12]},
            }
        }

        plan = _sample_replay_plan(sample)
        normalized = _apply_pipeline_mode(
            full_plan=plan,
            prompt_text="周期增强",
            ts_length=32,
            mode="replay_plan",
        )

        self.assertEqual(normalized["tool_name"], "season_enhance")
        self.assertEqual(normalized["parameters"]["region"], [4, 12])

    def test_runtime_strength_scalar_can_differ_from_benchmark_sort_scalar(self):
        sample = {"strength_label": 1, "strength_scalar": 0.5, "runtime_strength_scalar": 1.0}
        plan = {"parameters": {}}

        if "strength_label" in sample:
            plan["parameters"]["strength_label"] = sample["strength_label"]
        if "strength_scalar" in sample:
            plan["parameters"]["strength_scalar"] = sample["strength_scalar"]
        if "runtime_strength_scalar" in sample:
            plan["parameters"]["strength_scalar"] = sample["runtime_strength_scalar"]

        self.assertEqual(sample["strength_scalar"], 0.5)
        self.assertEqual(plan["parameters"]["strength_scalar"], 1.0)

    def test_strength_summary_verifier_rejects_nonmonotonic_family(self):
        payload = {
            "family_rows": [
                {
                    "family": "trend",
                    "primary_strength_metric": "edit_gain",
                    "weak_primary_strength_value": 1.0,
                    "medium_primary_strength_value": 0.5,
                    "strong_primary_strength_value": 2.0,
                    "primary_min_adjacent_gap_mean": -0.5,
                    "primary_monotonic_hit": 0.0,
                    "bg_mae_strong_minus_weak": 0.0,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            summary_path.write_text(json.dumps(payload), encoding="utf-8")
            result = verify_summary(
                str(summary_path),
                require_families=["trend"],
                min_primary_gap=0.0,
                max_bg_drift=1.0e-6,
                max_season_period_error=1.0e-6,
                max_season_level_drift=None,
                max_season_trend_drift=None,
            )

        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("not strictly monotonic" in item for item in result["failures"]))

    def test_replay_benchmark_builder_adds_plan_and_trend_runtime_scalar(self):
        payload = {
            "samples": [
                {
                    "sample_id": "trend_weak",
                    "base_ts": [1.0, 2.0],
                    "region": [0, 2],
                    "strength_label": 1,
                    "strength_scalar": 0.5,
                    "strength_text": "medium",
                    "duration_bucket": "short",
                    "edit_intent_gt": {"effect_family": "trend", "direction": "up", "shape": "linear"},
                },
                {
                    "sample_id": "season_weak",
                    "base_ts": [1.0, 2.0],
                    "region": [0, 2],
                    "strength_label": 0,
                    "strength_scalar": 0.0,
                    "strength_text": "weak",
                    "duration_bucket": "medium",
                    "edit_intent_gt": {"effect_family": "seasonality", "direction": "neutral", "shape": "periodic"},
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "benchmark.json"
            out = Path(tmpdir) / "replay.json"
            src.write_text(json.dumps(payload), encoding="utf-8")
            result = build_replay_benchmark(
                benchmark_path=str(src),
                output_path=str(out),
                trend_runtime_scalar="legacy_label",
            )
            replay = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(result["num_samples"], 2)
        self.assertEqual(replay["samples"][0]["replay_plan"]["tool_name"], "hybrid_up")
        self.assertEqual(replay["samples"][0]["runtime_strength_scalar"], 1.0)
        self.assertEqual(replay["samples"][1]["replay_plan"]["tool_name"], "season_enhance")

    def test_resolve_execution_instruction_falls_back_to_builder(self):
        plan = {
            "intent": {"effect_family": "seasonality", "shape": "periodic", "strength": "weak", "duration": "medium"},
            "execution": {},
            "parameters": {},
        }
        self.assertEqual(_resolve_execution_instruction(plan), "Apply a weak seasonality enhancement in this medium window")

    def test_season_enhance_target_attrs_are_strength_binned(self):
        self.assertTrue(np.array_equal(_resolve_season_enhance_tgt_attrs(strength_label=0), np.array([0, 0, 1], dtype=np.int64)))
        self.assertTrue(np.array_equal(_resolve_season_enhance_tgt_attrs(strength_label=1), np.array([0, 0, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(_resolve_season_enhance_tgt_attrs(strength_label=2), np.array([0, 0, 3], dtype=np.int64)))
        self.assertTrue(np.array_equal(_resolve_season_enhance_tgt_attrs(strength_scalar=0.0), np.array([0, 0, 1], dtype=np.int64)))
        self.assertTrue(np.array_equal(_resolve_season_enhance_tgt_attrs(strength_scalar=0.5), np.array([0, 0, 2], dtype=np.int64)))
        self.assertTrue(np.array_equal(_resolve_season_enhance_tgt_attrs(strength_scalar=1.0), np.array([0, 0, 3], dtype=np.int64)))

    def test_seasonality_amplitude_gain_is_strength_ordered(self):
        self.assertAlmostEqual(_resolve_seasonality_amplitude_gain(strength_label=0), 0.08)
        self.assertAlmostEqual(_resolve_seasonality_amplitude_gain(strength_label=1), 0.25)
        self.assertAlmostEqual(_resolve_seasonality_amplitude_gain(strength_label=2), 0.45)
        self.assertAlmostEqual(_resolve_seasonality_amplitude_gain(strength_scalar=0.0), 0.08)
        self.assertAlmostEqual(_resolve_seasonality_amplitude_gain(strength_scalar=0.5), 0.25)
        self.assertAlmostEqual(_resolve_seasonality_amplitude_gain(strength_scalar=1.0), 0.45)

    def test_seasonality_amplitude_anchor_grows_with_strength(self):
        ts = np.sin(np.linspace(0.0, 4.0 * np.pi, 96, endpoint=False)).astype(np.float32)
        weak_anchor = _build_seasonality_amplitude_anchor(ts, 16, 80, strength_label=0, apply_soft_mask=False)
        medium_anchor = _build_seasonality_amplitude_anchor(ts, 16, 80, strength_label=1, apply_soft_mask=False)
        strong_anchor = _build_seasonality_amplitude_anchor(ts, 16, 80, strength_label=2, apply_soft_mask=False)

        weak_gain = float(np.mean(np.abs(weak_anchor[16:80])))
        medium_gain = float(np.mean(np.abs(medium_anchor[16:80])))
        strong_gain = float(np.mean(np.abs(strong_anchor[16:80])))

        self.assertLess(weak_gain, medium_gain)
        self.assertLess(medium_gain, strong_gain)

    def test_seasonality_scaffold_delta_grows_with_strength(self):
        ts = np.sin(np.linspace(0.0, 4.0 * np.pi, 96, endpoint=False)).astype(np.float32)
        weak = _build_seasonality_scaffold(ts, 16, 80, strength_label=0, apply_soft_mask=False)
        medium = _build_seasonality_scaffold(ts, 16, 80, strength_label=1, apply_soft_mask=False)
        strong = _build_seasonality_scaffold(ts, 16, 80, strength_label=2, apply_soft_mask=False)

        weak_gain = float(np.mean(np.abs(weak[16:80] - ts[16:80])))
        medium_gain = float(np.mean(np.abs(medium[16:80] - ts[16:80])))
        strong_gain = float(np.mean(np.abs(strong[16:80] - ts[16:80])))

        self.assertLess(weak_gain, medium_gain)
        self.assertLess(medium_gain, strong_gain)

    def test_seasonality_fixed_period_metric_separates_amplitude_from_level(self):
        n = 96
        cycles = 4
        t = np.arange(n, dtype=np.float64)
        base = np.zeros(n, dtype=np.float64)
        target = np.sin(2.0 * np.pi * cycles * t / n)
        level_cheat = target + 5.0
        mask = np.ones(n, dtype=bool)
        config = {"cycles": cycles, "expected_period": n / cycles}

        target_metrics = strength_effect_eval._seasonality_frequency_fixed_metrics(base, target, target, mask, config)
        cheat_metrics = strength_effect_eval._seasonality_frequency_fixed_metrics(base, target, level_cheat, mask, config)

        self.assertAlmostEqual(target_metrics["target_fixed_period_fourier_amplitude"], 1.0, places=5)
        self.assertAlmostEqual(cheat_metrics["pred_fixed_period_fourier_amplitude"], 1.0, places=5)
        self.assertGreater(cheat_metrics["level_drift"], 4.9)

    def test_seasonality_fixed_period_metric_flags_frequency_cheat(self):
        n = 96
        expected_cycles = 4
        t = np.arange(n, dtype=np.float64)
        base = np.zeros(n, dtype=np.float64)
        target = np.sin(2.0 * np.pi * expected_cycles * t / n)
        frequency_cheat = np.sin(2.0 * np.pi * 8 * t / n)
        mask = np.ones(n, dtype=bool)
        config = {"cycles": expected_cycles, "expected_period": n / expected_cycles}

        metrics = strength_effect_eval._seasonality_frequency_fixed_metrics(base, target, frequency_cheat, mask, config)

        self.assertLess(metrics["pred_fixed_period_fourier_amplitude"], 0.05)
        self.assertGreater(metrics["dominant_period_error"], 10.0)

    def test_seasonality_dominant_period_uses_edit_delta_not_source_shape(self):
        n = 96
        expected_cycles = 4
        t = np.arange(n, dtype=np.float64)
        base = 3.0 * np.sin(2.0 * np.pi * 1 * t / n)
        seasonal_delta = np.sin(2.0 * np.pi * expected_cycles * t / n)
        target = base + seasonal_delta
        edited = target.copy()
        mask = np.ones(n, dtype=bool)
        config = {"cycles": expected_cycles, "expected_period": n / expected_cycles}

        metrics = strength_effect_eval._seasonality_frequency_fixed_metrics(base, target, edited, mask, config)

        self.assertAlmostEqual(metrics["dominant_period_error"], 0.0, places=6)

    def test_fixed_period_seasonality_scaffold_matches_config_frequency(self):
        n = 96
        start, end = 10, 58
        cycles = 4
        amplitude = 0.3
        phase = 0.4
        ts = np.zeros(n, dtype=np.float32)
        config = {
            "control_axis": "seasonality_amplitude",
            "frequency_edit_allowed": False,
            "cycles": cycles,
            "phase": phase,
            "seasonal_amplitude": amplitude,
            "expected_period": (end - start) / cycles,
        }

        scaffold = _build_fixed_period_seasonality_scaffold(ts, start, end, config)
        mask = np.zeros(n, dtype=bool)
        mask[start:end] = True
        metrics = strength_effect_eval._seasonality_frequency_fixed_metrics(
            ts,
            scaffold,
            scaffold,
            mask,
            config,
        )

        self.assertIsNotNone(scaffold)
        self.assertAlmostEqual(metrics["dominant_period_error"], 0.0, places=6)
        self.assertGreater(metrics["pred_fixed_period_fourier_amplitude"], 0.0)
        self.assertAlmostEqual(metrics["level_drift"], 0.0, places=6)
        self.assertAlmostEqual(metrics["trend_drift"], 0.0, places=6)

    def test_trend_injection_scaffold_uses_configured_hump_amplitude(self):
        ts = np.ones(12, dtype=np.float32) * 5.0
        scaffold = _build_trend_injection_scaffold(
            ts,
            2,
            10,
            {"injection_type": "trend_injection", "amplitude": 4.0},
        )

        self.assertIsNotNone(scaffold)
        self.assertAlmostEqual(float(scaffold[0]), 5.0)
        self.assertAlmostEqual(float(scaffold[2]), 5.0)
        self.assertAlmostEqual(float(scaffold[9]), 5.0)
        self.assertGreater(float(np.max(scaffold[2:10] - ts[2:10])), 3.7)

    def test_multiplier_scaffold_uses_configured_ratio_and_ramp(self):
        ts = np.ones(12, dtype=np.float32) * 2.0
        scaffold = _build_multiplier_scaffold(
            ts,
            2,
            10,
            {"injection_type": "multiplier", "multiplier": 2.0, "ramp_out": 3},
        )

        self.assertIsNotNone(scaffold)
        self.assertAlmostEqual(float(scaffold[2]), 4.0)
        self.assertAlmostEqual(float(scaffold[6]), 4.0)
        self.assertAlmostEqual(float(scaffold[9]), 2.0)
        self.assertAlmostEqual(float(scaffold[0]), 2.0)

    def test_step_change_scaffold_uses_configured_magnitude_and_ramp(self):
        ts = np.ones(12, dtype=np.float32) * 5.0
        scaffold = _build_step_change_scaffold(
            ts,
            2,
            10,
            {"injection_type": "step_change", "magnitude": -3.0, "ramp_out": 3},
        )

        self.assertIsNotNone(scaffold)
        self.assertAlmostEqual(float(scaffold[2]), 2.0)
        self.assertAlmostEqual(float(scaffold[6]), 2.0)
        self.assertAlmostEqual(float(scaffold[9]), 5.0)
        self.assertAlmostEqual(float(scaffold[0]), 5.0)

    def test_hard_zero_scaffold_uses_configured_floor_and_ramp(self):
        ts = np.arange(12, dtype=np.float32)
        scaffold = _build_hard_zero_scaffold(
            ts,
            2,
            10,
            {"injection_type": "hard_zero", "floor_value": 1.5, "ramp": 2},
        )

        self.assertIsNotNone(scaffold)
        self.assertAlmostEqual(float(scaffold[2]), 2.0)
        self.assertAlmostEqual(float(scaffold[3]), 1.5)
        self.assertAlmostEqual(float(scaffold[6]), 1.5)
        self.assertAlmostEqual(float(scaffold[9]), 10.0)
        self.assertAlmostEqual(float(scaffold[0]), 0.0)

    def test_noise_injection_scaffold_scales_same_template_monotonically(self):
        ts = np.ones(16, dtype=np.float32) * 10.0
        template = [-1.0, 0.0, 1.0, 0.5, -0.5, 1.5, -1.5, 0.25]
        weak = _build_noise_injection_scaffold(
            ts,
            4,
            12,
            {
                "injection_type": "noise_injection",
                "noise_std_ratio": 0.5,
                "baseline_offset": 0.0,
                "region_noise_std": 2.0,
                "noise_template": template,
            },
        )
        strong = _build_noise_injection_scaffold(
            ts,
            4,
            12,
            {
                "injection_type": "noise_injection",
                "noise_std_ratio": 1.5,
                "baseline_offset": 0.0,
                "region_noise_std": 2.0,
                "noise_template": template,
            },
        )

        self.assertIsNotNone(weak)
        self.assertIsNotNone(strong)
        weak_delta = np.asarray(weak[4:12] - ts[4:12], dtype=np.float64)
        strong_delta = np.asarray(strong[4:12] - ts[4:12], dtype=np.float64)
        self.assertGreater(float(np.std(strong_delta)), float(np.std(weak_delta)))
        self.assertAlmostEqual(float(np.mean(weak_delta)), 0.0, places=6)

    def test_legacy_scalar_family_dataset_meta_and_sweep_contract(self):
        csv_path = REPO_ROOT / "test_scripts" / "data" / "ETTh1.csv"
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "trend_legacy_family"
            result = build_family_dataset(
                csv_path=str(csv_path),
                dataset_name="ETTh1",
                output_dir=str(output_dir),
                seq_len=96,
                random_seed=17,
                train_families=1,
                valid_families=1,
                test_families=1,
                injection_types=["trend_injection"],
                selector="trend_injection",
                scalar_scheme="legacy_0_1_2",
            )
            meta = json.loads((Path(result["output_dir"]) / "meta.json").read_text(encoding="utf-8"))
            test_payload = json.loads((Path(result["output_dir"]) / "test.json").read_text(encoding="utf-8"))

        self.assertEqual(meta["scalar_scheme"], "legacy_0_1_2")
        self.assertEqual(meta["strength_axis"]["anchor_mapping"], {"weak": 0.0, "medium": 1.0, "strong": 2.0})
        self.assertEqual(test_payload["scalar_scheme"], "legacy_0_1_2")
        self.assertEqual(test_payload["strength_axis"]["range"], [0.0, 2.0])
        family_scalars = [sample["strength_scalar"] for sample in test_payload["families"][0]["samples"]]
        self.assertEqual(family_scalars, [0.0, 1.0, 2.0])
        self.assertEqual(trend_monotonic_eval._default_sweep_for_scheme("legacy_0_1_2"), [0.0, 0.5, 1.0, 1.5, 2.0])

    def test_final_output_strength_mapping_disabled_is_identity(self):
        model = Diff_CSDI_MultiPatch_Weaver_Parallel.__new__(Diff_CSDI_MultiPatch_Weaver_Parallel)
        model.final_output_strength_mapping_cfg = {"enabled": False}
        model.latest_final_output_strength_mapping_order_loss = None
        output = torch.randn(3, 1, 2, 1)

        mapped, gain = model._apply_final_output_strength_mapping(
            output,
            strength_cond=torch.randn(3, 4),
            strength_scalar=torch.tensor([0.0, 1.0, 2.0]),
        )

        self.assertIsNone(gain)
        self.assertTrue(torch.allclose(mapped, output))

    def test_final_output_strength_mapping_scalar_prior_is_ordered(self):
        model = Diff_CSDI_MultiPatch_Weaver_Parallel.__new__(Diff_CSDI_MultiPatch_Weaver_Parallel)
        model.final_output_strength_mapping_cfg = {
            "enabled": True,
            "mode": "bounded_scalar_gain",
            "scalar_center": 1.0,
            "scalar_prior_scale": 0.1,
            "learned_max_delta": 0.0,
            "min_gain": 0.8,
            "max_gain": 1.2,
            "hidden_dim": 4,
            "gain_order_margin": 0.01,
            "gain_order_weight": 0.2,
            "gain_order_direction": "increasing",
        }
        model.final_output_strength_mapping_head = None
        output = torch.ones(3, 1, 2, 1)

        mapped, gain = model._apply_final_output_strength_mapping(
            output,
            strength_cond=None,
            strength_scalar=torch.tensor([0.0, 1.0, 2.0]),
        )

        self.assertTrue(torch.allclose(gain.view(-1), torch.tensor([0.9, 1.0, 1.1])))
        self.assertTrue(torch.allclose(mapped[:, 0, 0, 0], torch.tensor([0.9, 1.0, 1.1])))
        self.assertIsNotNone(model.latest_final_output_strength_mapping_order_loss)
        self.assertAlmostEqual(float(model.latest_final_output_strength_mapping_order_loss.item()), 0.0, places=6)

    def test_final_output_strength_mapping_edit_region_scope_gates_gain_only(self):
        model = Diff_CSDI_MultiPatch_Weaver_Parallel.__new__(Diff_CSDI_MultiPatch_Weaver_Parallel)
        model.final_output_strength_mapping_cfg = {
            "enabled": True,
            "mode": "bounded_scalar_gain",
            "scalar_center": 1.0,
            "scalar_prior_scale": 0.1,
            "learned_max_delta": 0.0,
            "min_gain": 0.8,
            "max_gain": 1.2,
            "hidden_dim": 4,
            "gain_order_margin": 0.0,
            "gain_order_weight": 0.0,
            "gain_order_direction": "increasing",
            "scope": "edit_region",
        }
        model.final_output_strength_mapping_head = None
        output = torch.ones(2, 1, 3, 1)
        edit_mask = torch.tensor([
            [[1.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0]],
        ])

        mapped, gain = model._apply_final_output_strength_mapping(
            output,
            strength_cond=None,
            strength_scalar=torch.tensor([0.0, 2.0]),
            final_strength_mask=edit_mask,
        )

        self.assertEqual(tuple(gain.shape), tuple(output.shape))
        self.assertTrue(torch.allclose(mapped[0, 0, :, 0], torch.tensor([0.9, 1.0, 0.9])))
        self.assertTrue(torch.allclose(mapped[1, 0, :, 0], torch.tensor([1.0, 1.1, 1.0])))

    def test_trend_standard_mask_routed_eval_uses_edit_time_series_not_soft_region(self):
        calls = []

        class FakeWrapper:
            def __init__(self, model_path, config_path, device):
                calls.append(("init", model_path, config_path, device))

            def set_edit_steps(self, steps):
                calls.append(("set_edit_steps", steps))

            def edit_region_soft(self, **kwargs):
                raise AssertionError("standard route must not call edit_region_soft")

            def edit_time_series(self, **kwargs):
                calls.append(("edit_time_series", kwargs))
                if kwargs["ts"].shape != (4,):
                    raise AssertionError(f"unexpected ts shape {kwargs['ts'].shape}")
                expected_mask = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
                if not np.array_equal(kwargs["edit_mask"], expected_mask):
                    raise AssertionError(f"unexpected edit_mask {kwargs['edit_mask']}")
                return kwargs["ts"].reshape(1, -1) + float(kwargs["strength_scalar"])

        benchmark = {
            "families": [
                {
                    "family_id": "family_0",
                    "tool_name": "trend_injection",
                    "duration_bucket": "short",
                    "samples": [
                        {
                            "source_ts": [0.0, 1.0, 1.0, 0.0],
                            "target_ts": [0.0, 1.1, 1.1, 0.0],
                            "mask_gt": [0, 1, 1, 0],
                            "region": [1, 3],
                            "instruction_text": "increase trend weak",
                            "strength_scalar": 0.0,
                            "direction": "up",
                        },
                        {
                            "source_ts": [0.0, 1.0, 1.0, 0.0],
                            "target_ts": [0.0, 1.2, 1.2, 0.0],
                            "mask_gt": [0, 1, 1, 0],
                            "region": [1, 3],
                            "instruction_text": "increase trend medium",
                            "strength_scalar": 0.5,
                            "direction": "up",
                        },
                        {
                            "source_ts": [0.0, 1.0, 1.0, 0.0],
                            "target_ts": [0.0, 1.3, 1.3, 0.0],
                            "mask_gt": [0, 1, 1, 0],
                            "region": [1, 3],
                            "instruction_text": "increase trend strong",
                            "strength_scalar": 1.0,
                            "direction": "up",
                        },
                    ],
                }
            ]
        }

        original_wrapper = trend_monotonic_eval.TEditWrapper
        trend_monotonic_eval.TEditWrapper = FakeWrapper
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                benchmark_path = Path(tmpdir) / "benchmark.json"
                output_path = Path(tmpdir) / "out.json"
                benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")

                payload = trend_monotonic_eval.run_eval(
                    benchmark_path=benchmark_path,
                    model_path="fake.pth",
                    config_path="fake.yaml",
                    output_path=output_path,
                    max_families=1,
                    edit_steps=10,
                    sampler="ddim",
                    seed=1234,
                    device="cpu",
                    smooth_radius=3.0,
                    bg_drift_threshold=0.05,
                    probe_json=None,
                    sweep_values=[0.0, 0.5, 1.0],
                    generation_route="standard",
                    eval_mask_routed=True,
                )
        finally:
            trend_monotonic_eval.TEditWrapper = original_wrapper

        self.assertEqual(payload["config"]["generation_route"], "standard")
        self.assertTrue(payload["config"]["eval_mask_routed"])
        self.assertEqual(len([call for call in calls if call[0] == "edit_time_series"]), 3)

    def test_final_output_strength_mapping_zero_init_learned_head_preserves_prior(self):
        model = Diff_CSDI_MultiPatch_Weaver_Parallel.__new__(Diff_CSDI_MultiPatch_Weaver_Parallel)
        nn.Module.__init__(model)
        model.final_output_strength_mapping_cfg = {
            "enabled": True,
            "mode": "bounded_scalar_gain",
            "scalar_center": 1.0,
            "scalar_prior_scale": 0.1,
            "learned_max_delta": 0.1,
            "min_gain": 0.8,
            "max_gain": 1.2,
            "hidden_dim": 4,
            "gain_order_margin": 0.0,
            "gain_order_weight": 0.0,
            "gain_order_direction": "increasing",
        }
        model.final_output_strength_mapping_head = nn.Sequential(nn.Linear(4, 4), nn.SiLU(), nn.Linear(4, 1))
        nn.init.zeros_(model.final_output_strength_mapping_head[-1].weight)
        nn.init.zeros_(model.final_output_strength_mapping_head[-1].bias)
        output = torch.ones(3, 1, 2, 1)

        mapped, gain = model._apply_final_output_strength_mapping(
            output,
            strength_cond=torch.randn(3, 4),
            strength_scalar=torch.tensor([0.0, 1.0, 2.0]),
        )

        self.assertTrue(torch.allclose(gain.view(-1), torch.tensor([0.9, 1.0, 1.1])))
        self.assertTrue(torch.allclose(mapped[:, 0, 0, 0], torch.tensor([0.9, 1.0, 1.1])))

    def test_final_output_strength_mapping_order_loss_penalizes_flat_gain(self):
        model = Diff_CSDI_MultiPatch_Weaver_Parallel.__new__(Diff_CSDI_MultiPatch_Weaver_Parallel)
        model.final_output_strength_mapping_cfg = {
            "enabled": True,
            "mode": "bounded_scalar_gain",
            "scalar_center": 1.0,
            "scalar_prior_scale": 0.0,
            "learned_max_delta": 0.0,
            "min_gain": 0.8,
            "max_gain": 1.2,
            "hidden_dim": 4,
            "gain_order_margin": 0.05,
            "gain_order_weight": 0.2,
            "gain_order_direction": "increasing",
        }
        gain = torch.ones(3, 1, 1, 1)

        loss = model._compute_final_output_strength_mapping_order_loss(
            gain,
            torch.tensor([0.0, 1.0, 2.0]),
        )

        self.assertIsNotNone(loss)
        self.assertAlmostEqual(float(loss.item()), 0.05, places=6)

    def test_final_output_strength_mapping_decreasing_order_accepts_inverse_gain(self):
        model = Diff_CSDI_MultiPatch_Weaver_Parallel.__new__(Diff_CSDI_MultiPatch_Weaver_Parallel)
        model.final_output_strength_mapping_cfg = {
            "enabled": True,
            "mode": "bounded_scalar_gain",
            "scalar_center": 1.0,
            "scalar_prior_scale": -0.04,
            "learned_max_delta": 0.0,
            "min_gain": 0.9,
            "max_gain": 1.1,
            "hidden_dim": 4,
            "gain_order_margin": 0.01,
            "gain_order_weight": 0.2,
            "gain_order_direction": "decreasing",
        }
        model.final_output_strength_mapping_head = None
        output = torch.ones(3, 1, 2, 1)

        mapped, gain = model._apply_final_output_strength_mapping(
            output,
            strength_cond=None,
            strength_scalar=torch.tensor([0.0, 1.0, 2.0]),
        )

        self.assertTrue(torch.allclose(gain.view(-1), torch.tensor([1.04, 1.0, 0.96])))
        self.assertTrue(torch.allclose(mapped[:, 0, 0, 0], torch.tensor([1.04, 1.0, 0.96])))
        self.assertIsNotNone(model.latest_final_output_strength_mapping_order_loss)
        self.assertAlmostEqual(float(model.latest_final_output_strength_mapping_order_loss.item()), 0.0, places=6)

    def test_final_output_strength_mapping_scalar_transform_is_final_only(self):
        model = Diff_CSDI_MultiPatch_Weaver_Parallel.__new__(Diff_CSDI_MultiPatch_Weaver_Parallel)
        model.final_output_strength_mapping_cfg = {
            "enabled": True,
            "mode": "bounded_scalar_gain",
            "scalar_center": 1.0,
            "scalar_prior_scale": -0.04,
            "learned_max_delta": 0.0,
            "min_gain": 0.9,
            "max_gain": 1.1,
            "hidden_dim": 4,
            "gain_order_margin": 0.01,
            "gain_order_weight": 0.2,
            "gain_order_direction": "decreasing",
            "scalar_transform": {
                "enabled": True,
                "scale": 2.0,
                "offset": 0.0,
                "name": "current_0_0p5_1_to_legacy_0_1_2",
            },
        }
        model.final_output_strength_mapping_head = None
        output = torch.ones(3, 1, 2, 1)
        original_scalar = torch.tensor([0.0, 0.5, 1.0])

        mapped, gain = model._apply_final_output_strength_mapping(
            output,
            strength_cond=None,
            strength_scalar=original_scalar,
        )

        self.assertTrue(torch.equal(original_scalar, torch.tensor([0.0, 0.5, 1.0])))
        self.assertTrue(torch.allclose(gain.view(-1), torch.tensor([1.04, 1.0, 0.96])))
        self.assertTrue(torch.allclose(mapped[:, 0, 0, 0], torch.tensor([1.04, 1.0, 0.96])))
        self.assertEqual(model.latest_final_output_strength_mapping_scalar_transform["name"], "current_0_0p5_1_to_legacy_0_1_2")
        self.assertTrue(torch.allclose(
            model.latest_final_output_strength_mapping_scalar_transform["original_scalar"],
            torch.tensor([0.0, 0.5, 1.0]),
        ))
        self.assertTrue(torch.allclose(
            model.latest_final_output_strength_mapping_scalar_transform["transformed_scalar"],
            torch.tensor([0.0, 1.0, 2.0]),
        ))

    def test_final_output_strength_mapping_auto_legacy_half_step_transform(self):
        model = Diff_CSDI_MultiPatch_Weaver_Parallel.__new__(Diff_CSDI_MultiPatch_Weaver_Parallel)
        model.final_output_strength_mapping_cfg = {
            "enabled": True,
            "mode": "bounded_scalar_gain",
            "scalar_center": 1.0,
            "scalar_prior_scale": 0.04,
            "learned_max_delta": 0.0,
            "min_gain": 0.9,
            "max_gain": 1.1,
            "hidden_dim": 4,
            "gain_order_margin": 0.0,
            "gain_order_weight": 0.0,
            "gain_order_direction": "increasing",
            "scalar_transform": {
                "enabled": False,
                "scale": 1.0,
                "offset": 0.0,
                "name": "identity",
                "auto_legacy_half_step": True,
            },
        }
        model.final_output_strength_mapping_head = None
        output = torch.ones(3, 1, 2, 1)

        mapped, gain = model._apply_final_output_strength_mapping(
            output,
            strength_cond=None,
            strength_scalar=torch.tensor([0.0, 0.5, 1.0]),
        )

        self.assertTrue(torch.allclose(gain.view(-1), torch.tensor([0.96, 1.0, 1.04])))
        self.assertTrue(torch.allclose(mapped[:, 0, 0, 0], torch.tensor([0.96, 1.0, 1.04])))
        self.assertEqual(
            model.latest_final_output_strength_mapping_scalar_transform["name"],
            "auto_legacy_half_step_0_0p5_1_to_legacy_0_1_2",
        )
        self.assertTrue(torch.allclose(
            model.latest_final_output_strength_mapping_scalar_transform["transformed_scalar"],
            torch.tensor([0.0, 1.0, 2.0]),
        ))

    def test_final_output_strength_scale_is_identity_safe_and_scalar_ordered(self):
        model = Diff_CSDI_MultiPatch_Weaver_Parallel.__new__(Diff_CSDI_MultiPatch_Weaver_Parallel)
        model.final_output_strength_scale_cfg = {
            "enabled": True,
            "mode": "linear_scalar",
            "scale_per_unit": 0.2,
            "scalar_center": 1.0,
            "min_gain": 0.8,
            "max_gain": 1.2,
        }
        output = torch.ones(3, 1, 2, 1)

        scaled, gain = model._apply_final_output_strength_scale(
            output,
            strength_scalar=torch.tensor([0.0, 1.0, 2.0]),
        )

        self.assertTrue(torch.allclose(gain.view(-1), torch.tensor([0.8, 1.0, 1.2])))
        self.assertTrue(torch.allclose(scaled[:, 0, 0, 0], torch.tensor([0.8, 1.0, 1.2])))

    def test_final_output_strength_scale_disabled_is_identity(self):
        model = Diff_CSDI_MultiPatch_Weaver_Parallel.__new__(Diff_CSDI_MultiPatch_Weaver_Parallel)
        model.final_output_strength_scale_cfg = {"enabled": False}
        output = torch.randn(3, 1, 2, 1)

        scaled, gain = model._apply_final_output_strength_scale(
            output,
            strength_scalar=torch.tensor([0.0, 1.0, 2.0]),
        )

        self.assertIsNone(gain)
        self.assertTrue(torch.allclose(scaled, output))

    def test_tedit_wrapper_edit_mask_blends_background_back_to_source(self):
        wrapper = TEditWrapper(device="cpu")
        wrapper.is_loaded = True

        class FakeModel:
            edit_steps = 10

            def generate(self, batch, n_samples, mode="edit", sampler="ddim", return_diagnostics=False):
                src_x = batch["src_x"].permute(0, 2, 1)
                pred = src_x + 5.0
                sample = pred.unsqueeze(0)
                if not return_diagnostics:
                    return sample
                diagnostics = [{
                    "raw_reverse_output": pred.detach().cpu(),
                    "blended_output": pred.detach().cpu(),
                    "final_output": pred.detach().cpu(),
                    "raw_edit_region_mean_abs_delta": 5.0,
                    "final_edit_region_mean_abs_delta": 5.0,
                    "raw_background_mean_abs_delta": 5.0,
                    "final_background_mean_abs_delta": 5.0,
                    "blend_gap_edit_region_mean_abs": 0.0,
                    "blend_gap_background_mean_abs": 0.0,
                }]
                return sample, diagnostics

        wrapper.model = FakeModel()

        ts = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        mask = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
        src_attrs = np.array([0, 0, 0], dtype=np.int64)
        tgt_attrs = np.array([0, 0, 1], dtype=np.int64)

        edited, diagnostics = wrapper.edit_time_series(
            ts=ts,
            src_attrs=src_attrs,
            tgt_attrs=tgt_attrs,
            edit_mask=mask,
            return_diagnostics=True,
        )

        self.assertTrue(np.allclose(np.asarray(edited[0], dtype=np.float32), np.array([1.0, 7.0, 8.0, 4.0], dtype=np.float32)))
        self.assertAlmostEqual(diagnostics["model"][0]["final_background_mean_abs_delta"], 0.0, places=6)
        self.assertAlmostEqual(diagnostics["model"][0]["final_edit_region_mean_abs_delta"], 5.0, places=6)

    def test_finetuner_spacing_selection_uses_predicted_not_target_metrics(self):
        finetuner = Finetuner.__new__(Finetuner)
        summary = {
            "spacing_metrics_primary": True,
            "local_path_default_definition": "edit_time_series + edit_mask + final_output_strength_mapping.scope=edit_region",
            "raw_weak_le_medium_pass_rate": 1.0,
            "raw_medium_le_strong_pass_rate": 1.0,
            "raw_min_adjacent_gap_mean": 0.2,
            "raw_adjacent_gap_collapse_mean": 0.0,
            "raw_medium_minus_weak_mean": 0.2,
            "raw_strong_minus_medium_mean": 0.2,
            "raw_duration_bucket_spacing": {
                "medium": {
                    "min_adjacent_gap_mean": 0.3,
                    "adjacent_gap_collapse_mean": 0.0,
                },
                "long": {
                    "min_adjacent_gap_mean": 0.1,
                    "adjacent_gap_collapse_mean": 0.0,
                }
            },
            "final_weak_le_medium_pass_rate": 1.0,
            "final_medium_le_strong_pass_rate": 1.0,
            "final_min_adjacent_gap_mean": 0.2,
            "final_adjacent_gap_collapse_mean": 0.0,
            "final_duration_bucket_spacing": {
                "medium": {
                    "min_adjacent_gap_mean": 0.3,
                    "adjacent_gap_collapse_mean": 0.0,
                },
                "long": {
                    "min_adjacent_gap_mean": 0.1,
                    "adjacent_gap_collapse_mean": 0.0,
                },
            },
            "target_weak_le_medium_pass_rate": 0.0,
            "target_medium_le_strong_pass_rate": 0.0,
            "target_min_adjacent_gap_mean": -1.0,
            "target_adjacent_gap_collapse_mean": 1.0,
            "target_medium_minus_weak_mean": -1.0,
            "target_strong_minus_medium_mean": -1.0,
        }

        selection = finetuner._build_spacing_selection_payload(summary)

        self.assertAlmostEqual(selection["selection_score"], 1.625, places=6)
        self.assertAlmostEqual(selection["selection_score_v2"], 1.625, places=6)
        self.assertEqual(selection["selection_primary_domain"], "raw_local_spacing")
        self.assertEqual(selection["selection_tiebreak"], "collapse_then_loss")
        self.assertAlmostEqual(selection["weak_le_medium_pass_rate"], 1.0, places=6)
        self.assertAlmostEqual(selection["min_adjacent_gap_mean"], 0.2, places=6)
        self.assertAlmostEqual(selection["medium_bucket_min_adjacent_gap_mean"], 0.3, places=6)
        self.assertAlmostEqual(selection["long_bucket_min_adjacent_gap_mean"], 0.1, places=6)
        self.assertAlmostEqual(selection["target_spacing_reference"]["min_adjacent_gap_mean"], -1.0, places=6)
        self.assertAlmostEqual(selection["final_spacing_summary"]["min_adjacent_gap_mean"], 0.2, places=6)

    def test_finetuner_spacing_selection_tiebreaks_by_collapse_before_loss(self):
        finetuner = Finetuner.__new__(Finetuner)

        self.assertTrue(
            finetuner._is_better_selection(
                candidate_score=1.0,
                candidate_loss=20.0,
                best_score=1.0,
                best_loss=10.0,
                candidate_collapse=0.2,
                best_collapse=0.4,
            )
        )
        self.assertFalse(
            finetuner._is_better_selection(
                candidate_score=1.0,
                candidate_loss=5.0,
                best_score=1.0,
                best_loss=10.0,
                candidate_collapse=0.5,
                best_collapse=0.2,
            )
        )

    def test_finetuner_valid_summary_aggregates_raw_and_final_bucket_spacing(self):
        finetuner = Finetuner.__new__(Finetuner)
        predicted_diagnostics = [
            {
                "pred_duration_bucket_spacing": {
                    "medium": {"min_adjacent_gap_mean": 0.2, "adjacent_gap_collapse_mean": 0.5},
                    "long": {"min_adjacent_gap_mean": 0.1, "adjacent_gap_collapse_mean": 0.4},
                },
                "raw_duration_bucket_spacing": {
                    "medium": {"min_adjacent_gap_mean": 0.2, "adjacent_gap_collapse_mean": 0.5},
                    "long": {"min_adjacent_gap_mean": 0.1, "adjacent_gap_collapse_mean": 0.4},
                },
                "final_duration_bucket_spacing": {
                    "medium": {"min_adjacent_gap_mean": 0.25, "adjacent_gap_collapse_mean": 0.45},
                    "long": {"min_adjacent_gap_mean": 0.15, "adjacent_gap_collapse_mean": 0.35},
                },
                "raw_weak_le_medium_pass_rate": 1.0,
                "raw_medium_le_strong_pass_rate": 1.0,
                "raw_min_adjacent_gap_mean": 0.2,
                "raw_adjacent_gap_collapse_mean": 0.0,
                "final_weak_le_medium_pass_rate": 1.0,
                "final_medium_le_strong_pass_rate": 1.0,
                "final_min_adjacent_gap_mean": 0.2,
                "final_adjacent_gap_collapse_mean": 0.0,
                "spacing_metrics_primary": True,
            }
        ]

        predicted_summary = finetuner._summarize_numeric_dicts(predicted_diagnostics)
        predicted_summary["pred_duration_bucket_spacing"] = finetuner._aggregate_duration_bucket_spacing(predicted_diagnostics, "pred_duration_bucket_spacing")
        predicted_summary["raw_duration_bucket_spacing"] = finetuner._aggregate_duration_bucket_spacing(predicted_diagnostics, "raw_duration_bucket_spacing")
        predicted_summary["final_duration_bucket_spacing"] = finetuner._aggregate_duration_bucket_spacing(predicted_diagnostics, "final_duration_bucket_spacing")
        selection = finetuner._build_spacing_selection_payload(predicted_summary)

        self.assertAlmostEqual(selection["selection_score_v2"], 1.55, places=6)
        self.assertAlmostEqual(selection["raw_duration_bucket_spacing"]["medium"]["min_adjacent_gap_mean"], 0.2, places=6)
        self.assertAlmostEqual(selection["final_duration_bucket_spacing"]["long"]["min_adjacent_gap_mean"], 0.15, places=6)

    def test_finetuner_spacing_best_window_qualifies_when_score_and_medium_long_gap_are_positive(self):
        finetuner = Finetuner.__new__(Finetuner)
        finetuner.spacing_best_window_require_positive_score = True
        finetuner.spacing_best_window_require_positive_medium_long_gap = True
        selection = {
            "selection_score_v2": 0.1,
            "medium_bucket_min_adjacent_gap_mean": 0.01,
            "long_bucket_min_adjacent_gap_mean": 0.02,
        }
        self.assertTrue(finetuner._spacing_best_window_qualifies(selection))
        selection["long_bucket_min_adjacent_gap_mean"] = -0.01
        self.assertFalse(finetuner._spacing_best_window_qualifies(selection))

    def test_finetuner_spacing_early_stop_triggers_after_patience_without_refresh(self):
        finetuner = Finetuner.__new__(Finetuner)
        finetuner.spacing_best_window_require_positive_score = True
        finetuner.spacing_best_window_require_positive_medium_long_gap = True
        finetuner._best_valid_selection = {
            "spacing_selection": {
                "selection_score_v2": 0.5,
                "medium_bucket_min_adjacent_gap_mean": 0.3,
                "long_bucket_min_adjacent_gap_mean": 0.2,
            }
        }
        finetuner.output_folder = tempfile.mkdtemp()
        finetuner._spacing_early_stop_state = {
            "enabled": True,
            "patience": 2,
            "best_window_active": False,
            "best_window_enter_epoch": None,
            "no_improve_count": 0,
            "triggered": False,
            "trigger_epoch": None,
        }
        qualifying = {
            "selection_score_v2": 0.5,
            "medium_bucket_min_adjacent_gap_mean": 0.3,
            "long_bucket_min_adjacent_gap_mean": 0.2,
        }
        self.assertFalse(finetuner._update_spacing_early_stop_state(1, qualifying, is_best=True))
        self.assertTrue(finetuner._spacing_early_stop_state["best_window_active"])
        self.assertFalse(finetuner._update_spacing_early_stop_state(2, qualifying, is_best=False))
        self.assertEqual(finetuner._spacing_early_stop_state["no_improve_count"], 1)
        self.assertTrue(finetuner._update_spacing_early_stop_state(3, qualifying, is_best=False))
        self.assertTrue(finetuner._spacing_early_stop_state["triggered"])
        self.assertEqual(finetuner._spacing_early_stop_state["trigger_epoch"], 3)

    def test_discrete_strength_family_loader_preserves_duration_bucket(self):
        csv_path = REPO_ROOT / "test_scripts" / "data" / "ETTh1.csv"
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "trend_legacy_family"
            build_family_dataset(
                csv_path=str(csv_path),
                dataset_name="ETTh1",
                output_dir=str(output_dir),
                seq_len=96,
                random_seed=17,
                train_families=1,
                valid_families=1,
                test_families=1,
                injection_types=["trend_injection"],
                selector="trend_injection",
                scalar_scheme="legacy_0_1_2",
            )
            dataset = EditDataset({"name": "discrete_strength_family", "folder": str(output_dir)})
            batch = next(iter(dataset.get_loader(split="valid", batch_size=1, shuffle=False, include_self=False)))

        self.assertIsInstance(batch.get("duration_bucket"), list)
        self.assertEqual(len(batch["duration_bucket"]), 3)
        self.assertTrue(all(isinstance(item, str) and item for item in batch["duration_bucket"]))

    def test_strength_effect_eval_uses_edit_region_soft_local_route(self):
        calls = []

        class FakeWrapper:
            def __init__(self, model_path=None, config_path=None, device="cpu"):
                calls.append(("init", model_path, config_path, device))

            def load_model(self, model_path, config_path):
                calls.append(("load_model", model_path, config_path))

            def set_edit_steps(self, steps):
                calls.append(("set_edit_steps", steps))

            def edit_time_series(self, **kwargs):
                raise AssertionError("strength effect eval must not call edit_time_series")

            def edit_region_soft(self, **kwargs):
                calls.append(("edit_region_soft", kwargs))
                ts = np.asarray(kwargs["ts"], dtype=np.float32)
                edited = ts.copy()
                edited[:, 1:3] += np.asarray(kwargs["strength_scalar"], dtype=np.float32).reshape(-1, 1)
                diagnostics = {
                    "model": [{
                        "raw_reverse_output": torch.as_tensor(edited[:, None, :], dtype=torch.float32),
                    }],
                    "projector": [],
                    "modulation_base": [],
                    "modulation_weaver": [],
                    "generator": [],
                }
                return edited, diagnostics

        fake_records = [{
            "sample_idx": 0,
            "family_id": "family_0",
            "tool_name": "seasonality_injection",
            "family_semantic_tag": "seasonality",
            "duration_bucket": "short",
            "task_id": 0,
            "region_start": 1,
            "region_end": 3,
            "region_len": 2,
            "series_length": 4,
            "base": np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
            "src_attrs": np.array([0, 0, 0], dtype=np.int64),
            "tgt_attrs": np.array([0, 0, 2], dtype=np.int64),
            "edit_mask": np.array([False, True, True, False]),
            "controls": [
                {"strength_label": 0, "strength_scalar": 0.0, "strength_text": "weak", "instruction_text": "weak", "target": np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)},
                {"strength_label": 1, "strength_scalar": 0.5, "strength_text": "medium", "instruction_text": "medium", "target": np.array([0.0, 1.5, 1.5, 0.0], dtype=np.float32)},
                {"strength_label": 2, "strength_scalar": 1.0, "strength_text": "strong", "instruction_text": "strong", "target": np.array([0.0, 2.0, 2.0, 0.0], dtype=np.float32)},
            ],
        }]

        original_wrapper = strength_effect_eval.TEditWrapper
        original_resolve = strength_effect_eval._resolve_runtime_config_path
        original_load_config = strength_effect_eval._load_wrapper_config
        original_load_records = strength_effect_eval._load_eval_records
        original_argv = sys.argv[:]
        strength_effect_eval.TEditWrapper = FakeWrapper
        strength_effect_eval._resolve_runtime_config_path = lambda model_path, config_path: config_path
        strength_effect_eval._load_wrapper_config = lambda config_path: {"attrs": {}, "side": {}, "diffusion": {}}
        strength_effect_eval._load_eval_records = lambda dataset_folder, split, max_samples: (fake_records, ["trend_types", "trend_directions", "season_cycles"])
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "eval.json"
                sys.argv = [
                    "evaluate_tedit_strength_effect.py",
                    "--model-path", "fake.pth",
                    "--config-path", "fake.yaml",
                    "--dataset-folder", tmpdir,
                    "--split", "test",
                    "--max-samples", "1",
                    "--device", "cpu",
                    "--edit-steps", "9",
                    "--output", str(output_path),
                ]
                strength_effect_eval.main()
                payload = json.loads(output_path.read_text(encoding="utf-8"))
        finally:
            strength_effect_eval.TEditWrapper = original_wrapper
            strength_effect_eval._resolve_runtime_config_path = original_resolve
            strength_effect_eval._load_wrapper_config = original_load_config
            strength_effect_eval._load_eval_records = original_load_records
            sys.argv = original_argv

        edit_calls = [call for call in calls if call[0] == "edit_region_soft"]
        self.assertEqual(len(edit_calls), 1)
        self.assertEqual(edit_calls[0][1]["start_idx"], 1)
        self.assertEqual(edit_calls[0][1]["end_idx"], 3)
        self.assertEqual(edit_calls[0][1]["strength_scalar"], [0.0, 0.5, 1.0])
        self.assertEqual(payload["config"]["generation_route"], "soft_region")
        self.assertEqual(payload["config"]["generation_route_label"], "soft_local_edit_region")
        self.assertEqual(payload["config"]["acceptance_route"], "local_path_soft_mask")
        self.assertEqual(payload["summary"]["route_statement"], "edit_region_soft + soft local mask + latent blending")

    def test_strength_effect_eval_supports_trend_standard_mask_routed_route(self):
        calls = []

        class FakeWrapper:
            def __init__(self, model_path=None, config_path=None, device="cpu"):
                calls.append(("init", model_path, config_path, device))

            def load_model(self, model_path, config_path):
                calls.append(("load_model", model_path, config_path))

            def set_edit_steps(self, steps):
                calls.append(("set_edit_steps", steps))

            def edit_region_soft(self, **kwargs):
                raise AssertionError("trend standard route must not call edit_region_soft")

            def edit_time_series(self, **kwargs):
                calls.append(("edit_time_series", kwargs))
                ts = np.asarray(kwargs["ts"], dtype=np.float32)
                expected_mask = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
                if not np.array_equal(np.asarray(kwargs["edit_mask"], dtype=np.float32), expected_mask):
                    raise AssertionError(f"unexpected edit_mask {kwargs['edit_mask']}")
                edited = ts.copy()
                edited[:, 1:3] += np.asarray(kwargs["strength_scalar"], dtype=np.float32).reshape(-1, 1)
                diagnostics = {
                    "model": [{
                        "raw_reverse_output": torch.as_tensor(edited[:, None, :], dtype=torch.float32),
                    }],
                    "projector": [],
                    "modulation_base": [],
                    "modulation_weaver": [],
                    "generator": [],
                }
                return edited, diagnostics

        fake_records = [{
            "sample_idx": 0,
            "family_id": "family_0",
            "tool_name": "trend_injection",
            "family_semantic_tag": "trend",
            "duration_bucket": "short",
            "task_id": 0,
            "region_start": 1,
            "region_end": 3,
            "region_len": 2,
            "series_length": 4,
            "base": np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
            "src_attrs": np.array([1, 0, 0], dtype=np.int64),
            "tgt_attrs": np.array([2, 1, 0], dtype=np.int64),
            "edit_mask": np.array([False, True, True, False]),
            "controls": [
                {"strength_label": 0, "strength_scalar": 0.0, "strength_text": "weak", "instruction_text": "weak", "target": np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)},
                {"strength_label": 1, "strength_scalar": 0.5, "strength_text": "medium", "instruction_text": "medium", "target": np.array([0.0, 1.5, 1.5, 0.0], dtype=np.float32)},
                {"strength_label": 2, "strength_scalar": 1.0, "strength_text": "strong", "instruction_text": "strong", "target": np.array([0.0, 2.0, 2.0, 0.0], dtype=np.float32)},
            ],
        }]

        original_wrapper = strength_effect_eval.TEditWrapper
        original_resolve = strength_effect_eval._resolve_runtime_config_path
        original_load_config = strength_effect_eval._load_wrapper_config
        original_load_records = strength_effect_eval._load_eval_records
        original_argv = sys.argv[:]
        strength_effect_eval.TEditWrapper = FakeWrapper
        strength_effect_eval._resolve_runtime_config_path = lambda model_path, config_path: config_path
        strength_effect_eval._load_wrapper_config = lambda config_path: {"attrs": {}, "side": {}, "diffusion": {}}
        strength_effect_eval._load_eval_records = lambda dataset_folder, split, max_samples: (fake_records, ["trend_types", "trend_directions", "season_cycles"])
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "eval.json"
                sys.argv = [
                    "evaluate_tedit_strength_effect.py",
                    "--model-path", "fake.pth",
                    "--config-path", "fake.yaml",
                    "--dataset-folder", tmpdir,
                    "--split", "test",
                    "--max-samples", "1",
                    "--device", "cpu",
                    "--edit-steps", "9",
                    "--generation-route", "standard",
                    "--eval-mask-routed", "1",
                    "--final-mapping-scope", "edit_region",
                    "--output", str(output_path),
                ]
                strength_effect_eval.main()
                payload = json.loads(output_path.read_text(encoding="utf-8"))
        finally:
            strength_effect_eval.TEditWrapper = original_wrapper
            strength_effect_eval._resolve_runtime_config_path = original_resolve
            strength_effect_eval._load_wrapper_config = original_load_config
            strength_effect_eval._load_eval_records = original_load_records
            sys.argv = original_argv

        edit_calls = [call for call in calls if call[0] == "edit_time_series"]
        self.assertEqual(len(edit_calls), 1)
        self.assertEqual(edit_calls[0][1]["edit_steps"], 9)
        self.assertEqual(edit_calls[0][1]["strength_scalar"], [0.0, 0.5, 1.0])
        self.assertEqual(payload["config"]["generation_route"], "standard")
        self.assertEqual(payload["config"]["generation_route_label"], "standard_edit_time_series")
        self.assertEqual(payload["config"]["acceptance_route"], "local_path_mask_routed")
        self.assertTrue(payload["config"]["eval_mask_routed"])
        self.assertEqual(payload["summary"]["route_statement"], "edit_time_series + edit_mask + final_output_strength_mapping.scope=edit_region")

    def test_carrier_mix_restores_residual_without_changing_skip(self):
        block = ResidualBlock(
            side_dim=4,
            attr_dim=8,
            channels=2,
            diffusion_embedding_dim=8,
            nheads=1,
            is_linear=False,
            strength_cond_dim=4,
            strength_mode="amplitude_decomposition",
            strength_gain_multiplier=1.0,
            is_attr_proj=False,
            output_branch_carrier={
                "enabled": True,
                "mode": "skip_residual_mix",
                "skip_scale": 0.25,
                "min_residual_to_skip_ratio": 0.5,
                "scalar_order_margin": 0.1,
            },
        )
        with torch.no_grad():
            block.mid_projection.weight.zero_()
            block.mid_projection.bias.zero_()
            block.side_projection.weight.zero_()
            block.side_projection.bias.zero_()
            block.output_projection.weight.zero_()
            block.output_projection.bias.zero_()
            block.output_projection.bias[:2].fill_(0.0)
            block.output_projection.bias[2:].fill_(4.0)

        residual, skip = block._run_output_head(
            y=torch.zeros(1, 2, 1, 3, dtype=torch.float32),
            side_emb=torch.zeros(1, 4, 1, 3, dtype=torch.float32),
            base_shape=(1, 2, 1, 3),
            strength_cond=None,
            strength_scalar=torch.tensor([0.0]),
        )

        self.assertTrue(torch.allclose(residual, torch.full_like(residual, 1.0)))
        self.assertTrue(torch.allclose(skip, torch.full_like(skip, 4.0)))
        self.assertIsNotNone(block._latest_output_branch_regularizer_loss)
        self.assertAlmostEqual(float(block._latest_output_branch_regularizer_loss.item()), 2.0, places=6)
        self.assertIsNone(block._latest_output_branch_scalar_order_loss)

    def test_scalar_order_loss_penalizes_flat_residual_amplitude(self):
        block = ResidualBlock(
            side_dim=4,
            attr_dim=8,
            channels=2,
            diffusion_embedding_dim=8,
            nheads=1,
            is_linear=False,
            strength_cond_dim=4,
            strength_mode="amplitude_decomposition",
            strength_gain_multiplier=1.0,
            is_attr_proj=False,
            output_branch_carrier={
                "enabled": False,
                "mode": "skip_residual_mix",
                "skip_scale": 0.0,
                "min_residual_to_skip_ratio": 0.0,
                "scalar_order_margin": 0.2,
            },
        )
        flat = torch.ones(3, 2, 1, 4)
        loss = block._compute_scalar_order_loss(flat, torch.tensor([0.0, 1.0, 2.0]))

        self.assertIsNotNone(loss)
        self.assertAlmostEqual(float(loss.item()), 0.2, places=6)

    def test_finetune_uses_main_edit_branch_loss_before_bootstrap_overwrite(self):
        model = ConditionalGenerator.__new__(ConditionalGenerator)
        model.device = torch.device("cpu")
        model.bootstrap_ratio = 0.5
        model.num_steps = 2
        model.diffusion_loss_weight = 0.0
        model.output_branch_regularizer_weight = 1.0
        model.output_branch_scalar_order_weight = 1.0
        model.final_output_strength_mapping_order_weight = 1.0
        model._latest_loss_breakdown = None
        model.diff_model = types.SimpleNamespace(
            latest_output_branch_regularizer_loss=None,
            latest_output_branch_scalar_order_loss=None,
            latest_final_output_strength_mapping_order_loss=None,
        )
        model.side_en = lambda tp: tp
        model.attr_en = lambda attrs: attrs.float()

        calls = {"count": 0}

        def fake_edit(src_x, *args, **kwargs):
            calls["count"] += 1
            value = 3.0 if calls["count"] == 1 else 99.0
            model.diff_model.latest_output_branch_regularizer_loss = src_x.new_tensor(value)
            model.diff_model.latest_output_branch_scalar_order_loss = src_x.new_tensor(value + 1.0)
            model.diff_model.latest_final_output_strength_mapping_order_loss = src_x.new_tensor(value + 2.0)
            return src_x + 1.0

        model._edit = fake_edit
        model._noise_estimation_loss = lambda *args, **kwargs: torch.tensor(0.0)

        batch = {
            "src_x": torch.zeros(2, 4, 1),
            "tgt_x": torch.ones(2, 4, 1),
            "tp": torch.zeros(2, 4, 1),
            "src_attrs": torch.zeros(2, 3, dtype=torch.long),
            "tgt_attrs": torch.ones(2, 3, dtype=torch.long),
        }

        loss = model.fintune(batch, is_train=True)

        self.assertEqual(calls["count"], 2)
        self.assertAlmostEqual(float(loss.item()), 12.0, places=6)
        self.assertAlmostEqual(model._latest_loss_breakdown["output_branch_regularizer_loss"], 3.0, places=6)
        self.assertAlmostEqual(model._latest_loss_breakdown["weighted_output_branch_regularizer_loss"], 3.0, places=6)
        self.assertAlmostEqual(model._latest_loss_breakdown["output_branch_scalar_order_loss"], 4.0, places=6)
        self.assertAlmostEqual(model._latest_loss_breakdown["weighted_output_branch_scalar_order_loss"], 4.0, places=6)
        self.assertAlmostEqual(model._latest_loss_breakdown["final_output_strength_mapping_order_loss"], 5.0, places=6)
        self.assertAlmostEqual(model._latest_loss_breakdown["weighted_final_output_strength_mapping_order_loss"], 5.0, places=6)


if __name__ == "__main__":
    unittest.main()
