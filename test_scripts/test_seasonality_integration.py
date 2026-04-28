import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_scripts.build_event_driven_testset import InjectorFactory
from test_scripts.build_tedit_strength_discrete_benchmark import build_discrete_benchmark
from test_scripts.build_tedit_strength_trend_family_dataset import build_family_dataset


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
        self.assertEqual([sample["strength_text"] for sample in family["samples"]], ["weak", "medium", "strong"])

    def test_family_dataset_meta_uses_season_cycle_control_axis(self):
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
        self.assertEqual(meta["control_attr"], ["season_cycles"])
        self.assertEqual(meta["control_attr_ids"], [2])


if __name__ == "__main__":
    unittest.main()
