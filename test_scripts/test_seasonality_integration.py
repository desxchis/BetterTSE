import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_scripts.build_event_driven_testset import InjectorFactory
from test_scripts.build_tedit_strength_discrete_benchmark import build_discrete_benchmark
from test_scripts.build_strength_pipeline_main_experiment_benchmark import build_pipeline_benchmark
from test_scripts.build_tedit_strength_trend_family_dataset import build_family_dataset
from test_scripts.evaluate_tedit_strength_effect import _load_eval_records
from test_scripts.summarize_strength_pipeline_main_experiment import _family_stats


REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "test_scripts" / "data" / "ETTh1.csv"


class TestSeasonalityIntegration(unittest.TestCase):
    def test_event_driven_factory_registers_seasonality_injection(self):
        factory = InjectorFactory(random_seed=7)
        injector = factory.create_injector("seasonality_injection")
        base_ts = np.linspace(0.0, 1.0, 96, dtype=np.float64)

        target_ts, mask_gt, config = injector.inject(base_ts, start_step=24, duration=36)
        intent = injector.get_edit_intent(config)

        self.assertEqual(injector.get_name(), "seasonality_injection")
        self.assertEqual(config["injection_type"], "seasonality_injection")
        self.assertIn(config["cycles"], {1, 2, 4})
        self.assertEqual(intent["effect_family"], "seasonality")
        self.assertEqual(intent["shape"], "periodic")
        self.assertEqual(int(np.sum(mask_gt)), 36)
        self.assertEqual(target_ts.shape, base_ts.shape)

    def test_discrete_benchmark_builds_seasonality_family(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = build_discrete_benchmark(
                csv_path=str(CSV_PATH),
                dataset_name="ETTh1",
                output_dir=tmpdir,
                num_families=1,
                seq_len=96,
                random_seed=11,
                injection_types=["seasonality_injection"],
            )
            payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

        self.assertEqual(payload["injection_types"], ["seasonality_injection"])
        self.assertEqual(len(payload["families"]), 1)
        family = payload["families"][0]
        self.assertEqual(family["tool_name"], "seasonality_injection")
        self.assertEqual(family["task_id"], 2)
        self.assertEqual(family["attr_strategy"], "native")
        self.assertIn(family["shape"], {"periodic", "flatten"})
        self.assertEqual(family["injection_config"]["seasonality_mode"], "amplitude_fixed_frequency")
        self.assertTrue(family["injection_config"]["fixed_period"])
        self.assertFalse(family["injection_config"]["frequency_edit_allowed"])
        self.assertEqual([sample["strength_text"] for sample in family["samples"]], ["weak", "medium", "strong"])
        cycles = [sample["injection_config"]["cycles"] for sample in family["samples"]]
        phases = [sample["injection_config"]["phase"] for sample in family["samples"]]
        self.assertEqual(len(set(cycles)), 1)
        self.assertEqual(len(set(phases)), 1)

    def test_family_dataset_meta_uses_seasonality_amplitude_control_axis(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "family_dataset"
            result = build_family_dataset(
                csv_path=str(CSV_PATH),
                dataset_name="ETTh1",
                output_dir=str(output_dir),
                seq_len=96,
                random_seed=13,
                train_families=1,
                valid_families=1,
                test_families=1,
                injection_types=["seasonality_injection"],
                selector="seasonality_injection",
            )
            meta = json.loads((Path(result["output_dir"]) / "meta.json").read_text(encoding="utf-8"))

        self.assertEqual(meta["selector"], "seasonality_injection")
        self.assertEqual(meta["control_attr"], ["seasonality_amplitude"])
        self.assertEqual(meta["control_attr_ids"], [])
        self.assertEqual(meta["control_definition"]["axis"], "seasonality_amplitude")
        self.assertIn("frequency", meta["control_definition"]["forbidden"])

    def test_strength_eval_loader_preserves_seasonality_frequency_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "family_dataset"
            result = build_family_dataset(
                csv_path=str(CSV_PATH),
                dataset_name="ETTh1",
                output_dir=str(output_dir),
                seq_len=96,
                random_seed=19,
                train_families=1,
                valid_families=1,
                test_families=1,
                injection_types=["seasonality_injection"],
                selector="seasonality_injection",
            )
            records, _ = _load_eval_records(result["output_dir"], "test", 1)

        config = records[0]["injection_config"]
        self.assertEqual(records[0]["tool_name"], "seasonality_injection")
        self.assertEqual(config["control_axis"], "seasonality_amplitude")
        self.assertTrue(config["fixed_period"])
        self.assertFalse(config["frequency_edit_allowed"])
        self.assertGreater(config["cycles"], 0)

    def test_pipeline_benchmark_preserves_seasonality_injection_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = build_pipeline_benchmark(
                csv_path=str(CSV_PATH),
                dataset_name="ETTh1",
                output_dir=tmpdir,
                num_families=2,
                seq_len=96,
                random_seed=23,
            )
            payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
        sample = next(row for row in payload["samples"] if row["edit_intent_gt"]["effect_family"] == "seasonality")
        self.assertEqual(sample["injection_config"]["control_axis"], "seasonality_amplitude")
        self.assertTrue(sample["injection_config"]["fixed_period"])
        self.assertFalse(sample["injection_config"]["frequency_edit_allowed"])
        self.assertEqual(sample["pipeline_options"]["family_region_source"], "benchmark")

    def test_pipeline_summary_uses_fourier_amplitude_as_seasonality_primary_metric(self):
        n = 96
        cycles = 4
        t = np.arange(n, dtype=np.float64)
        base = np.zeros(n, dtype=np.float64)
        region = [16, 80]
        region_t = np.arange(region[1] - region[0], dtype=np.float64)
        rows = []
        for label, strength_text, amp in [(0, "weak", 0.5), (1, "medium", 1.0), (2, "strong", 1.5)]:
            generated = base.copy()
            generated[region[0]:region[1]] = amp * np.sin(2.0 * np.pi * cycles * region_t / (region[1] - region[0]))
            row = {
                "sample_id": f"season_{strength_text}",
                "family_id": "season_family",
                "base_ts": base.tolist(),
                "target_ts": generated.tolist(),
                "generated_ts": generated.tolist(),
                "region": region,
                "strength_text": strength_text,
                "strength_label": label,
                "strength_scalar": float(label) / 2.0,
                "duration_bucket": "medium",
                "edit_intent_gt": {"effect_family": "seasonality", "direction": "neutral", "shape": "periodic"},
                "llm_plan": {"execution": {"route_type": "test"}, "tool_name": "season_enhance"},
                "intent_alignment": {"match_score": 1.0, "effect_family_match": 1.0, "shape_match": 1.0, "direction_match": 1.0},
                "metrics": {"preservation_mae": 0.0},
                "background_fidelity": {"max_err": 0.0},
                "injection_config": {
                    "control_axis": "seasonality_amplitude",
                    "fixed_period": True,
                    "frequency_edit_allowed": False,
                    "cycles": cycles,
                    "expected_period": (region[1] - region[0]) / cycles,
                },
            }
            rows.append(row)

        stats = _family_stats(rows)

        self.assertEqual(stats["primary_strength_metric"], "fixed_period_fourier_amplitude")
        self.assertLess(stats["weak_primary_strength_value"], stats["medium_primary_strength_value"])
        self.assertLess(stats["medium_primary_strength_value"], stats["strong_primary_strength_value"])
        self.assertEqual(stats["primary_monotonic_hit"], 1.0)
        self.assertAlmostEqual(stats["dominant_period_error_max"], 0.0, places=6)

    def test_seasonality_primary_metric_uses_edit_delta_not_source_amplitude(self):
        n = 128
        cycles = 2
        region = [16, 112]
        region_len = region[1] - region[0]
        t = np.arange(region_len, dtype=np.float64)
        base = np.zeros(n, dtype=np.float64)
        base[region[0]:region[1]] = -2.0 * np.sin(2.0 * np.pi * cycles * t / region_len)
        rows = []
        for label, strength_text, amp in [(0, "weak", 0.5), (1, "medium", 1.0), (2, "strong", 1.5)]:
            generated = base.copy()
            delta = amp * np.sin(2.0 * np.pi * cycles * t / region_len)
            generated[region[0]:region[1]] = base[region[0]:region[1]] + delta
            rows.append({
                "sample_id": f"season_antiphase_{strength_text}",
                "family_id": "season_antiphase_family",
                "base_ts": base.tolist(),
                "target_ts": generated.tolist(),
                "generated_ts": generated.tolist(),
                "region": region,
                "strength_text": strength_text,
                "strength_label": label,
                "strength_scalar": float(label) / 2.0,
                "duration_bucket": "medium",
                "edit_intent_gt": {"effect_family": "seasonality", "direction": "neutral", "shape": "periodic"},
                "llm_plan": {"execution": {"route_type": "test"}, "tool_name": "season_enhance", "parameters": {"region": region}},
                "intent_alignment": {"match_score": 1.0, "effect_family_match": 1.0, "shape_match": 1.0, "direction_match": 1.0},
                "metrics": {"preservation_mae": 0.0},
                "background_fidelity": {"max_err": 0.0},
                "injection_config": {
                    "control_axis": "seasonality_amplitude",
                    "fixed_period": True,
                    "frequency_edit_allowed": False,
                    "cycles": cycles,
                    "expected_period": region_len / cycles,
                },
            })

        stats = _family_stats(rows)

        self.assertEqual(stats["primary_strength_metric"], "fixed_period_fourier_amplitude")
        self.assertLess(stats["weak_primary_strength_value"], stats["medium_primary_strength_value"])
        self.assertLess(stats["medium_primary_strength_value"], stats["strong_primary_strength_value"])
        self.assertEqual(stats["primary_monotonic_hit"], 1.0)

    def test_pipeline_summary_uses_family_specific_primary_metrics_for_level_like_families(self):
        n = 32
        base = np.ones(n, dtype=np.float64) * 10.0
        region = [8, 24]

        cases = {
            "hard_zero": ("zero_suppression_delta", [8.0, 5.0, 2.0]),
            "step_change": ("step_level_shift", [11.0, 13.0, 16.0]),
            "multiplier": ("multiplicative_abs_ratio", [11.0, 14.0, 18.0]),
        }
        for family, (metric_name, levels) in cases.items():
            rows = []
            for label, strength_text, level in zip([0, 1, 2], ["weak", "medium", "strong"], levels):
                generated = base.copy()
                generated[region[0]:region[1]] = level
                rows.append({
                    "sample_id": f"{family}_{strength_text}",
                    "family_id": f"{family}_family",
                    "base_ts": base.tolist(),
                    "target_ts": generated.tolist(),
                    "generated_ts": generated.tolist(),
                    "region": region,
                    "strength_text": strength_text,
                    "strength_label": label,
                    "strength_scalar": float(label) / 2.0,
                    "duration_bucket": "medium",
                    "edit_intent_gt": {"effect_family": family, "direction": "neutral", "shape": "local"},
                    "llm_plan": {"execution": {"route_type": "test"}, "tool_name": family},
                    "intent_alignment": {"match_score": 1.0, "effect_family_match": 1.0, "shape_match": 1.0, "direction_match": 1.0},
                    "metrics": {"preservation_mae": 0.0},
                    "background_fidelity": {"max_err": 0.0},
                })
            stats = _family_stats(rows)
            self.assertEqual(stats["primary_strength_metric"], metric_name)
            self.assertEqual(stats["primary_monotonic_hit"], 1.0)

    def test_pipeline_summary_uses_noise_roughness_primary_metric(self):
        n = 32
        base = np.zeros(n, dtype=np.float64)
        region = [8, 24]
        pattern = np.tile(np.array([1.0, -1.0], dtype=np.float64), (region[1] - region[0]) // 2)
        rows = []
        for label, strength_text, amp in [(0, "weak", 0.2), (1, "medium", 0.6), (2, "strong", 1.2)]:
            generated = base.copy()
            generated[region[0]:region[1]] = amp * pattern
            rows.append({
                "sample_id": f"noise_{strength_text}",
                "family_id": "noise_family",
                "base_ts": base.tolist(),
                "target_ts": generated.tolist(),
                "generated_ts": generated.tolist(),
                "region": region,
                "strength_text": strength_text,
                "strength_label": label,
                "strength_scalar": float(label) / 2.0,
                "duration_bucket": "medium",
                "edit_intent_gt": {"effect_family": "noise_injection", "direction": "neutral", "shape": "irregular_noise"},
                "llm_plan": {"execution": {"route_type": "test"}, "tool_name": "noise_injection"},
                "intent_alignment": {"match_score": 1.0, "effect_family_match": 1.0, "shape_match": 1.0, "direction_match": 1.0},
                "metrics": {"preservation_mae": 0.0},
                "background_fidelity": {"max_err": 0.0},
            })

        stats = _family_stats(rows)

        self.assertEqual(stats["primary_strength_metric"], "local_noise_roughness_delta")
        self.assertEqual(stats["primary_monotonic_hit"], 1.0)


if __name__ == "__main__":
    unittest.main()
