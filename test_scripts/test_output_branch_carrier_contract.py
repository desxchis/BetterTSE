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
from train.finetuner import Finetuner
import test_scripts.evaluate_tedit_strength_effect as strength_effect_eval
import test_scripts.run_tedit_trend_monotonic_eval as trend_monotonic_eval
from test_scripts.build_tedit_strength_trend_family_dataset import build_family_dataset
from tool.tedit_wrapper import TEditWrapper


class TestOutputBranchCarrierContract(unittest.TestCase):
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
            "pred_weak_le_medium_pass_rate": 1.0,
            "pred_medium_le_strong_pass_rate": 1.0,
            "pred_min_adjacent_gap_mean": 0.2,
            "pred_adjacent_gap_collapse_mean": 0.0,
            "pred_medium_minus_weak_mean": 0.2,
            "pred_strong_minus_medium_mean": 0.2,
            "pred_duration_bucket_spacing": {
                "long": {
                    "min_adjacent_gap_mean": 0.1,
                    "adjacent_gap_collapse_mean": 0.0,
                }
            },
            "target_weak_le_medium_pass_rate": 0.0,
            "target_medium_le_strong_pass_rate": 0.0,
            "target_min_adjacent_gap_mean": -1.0,
            "target_adjacent_gap_collapse_mean": 1.0,
            "target_medium_minus_weak_mean": -1.0,
            "target_strong_minus_medium_mean": -1.0,
        }

        selection = finetuner._build_spacing_selection_payload(summary)

        self.assertAlmostEqual(selection["selection_score"], 1.2, places=6)
        self.assertAlmostEqual(selection["weak_le_medium_pass_rate"], 1.0, places=6)
        self.assertAlmostEqual(selection["min_adjacent_gap_mean"], 0.2, places=6)
        self.assertAlmostEqual(selection["target_spacing_reference"]["min_adjacent_gap_mean"], -1.0, places=6)

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
