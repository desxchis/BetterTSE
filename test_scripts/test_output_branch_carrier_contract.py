import types
import unittest
from pathlib import Path
import sys

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


class TestOutputBranchCarrierContract(unittest.TestCase):
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
