import unittest
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from bin.train import TrainConfig


class ConfigV2Test(unittest.TestCase):
    def test_model_matrix_is_complete_and_uses_central_output_paths(self):
        root = Path("configs/config_v2")
        expected = {
            root / representation / allocation / f"{size}.yaml"
            for representation in ("peak", "dense", "hybrid")
            for allocation in ("standard", "heavy")
            for size in ("small", "medium", "large")
        }

        self.assertEqual(set(root.glob("*/*/*.yaml")), expected)

        for path in sorted(expected):
            with self.subTest(path=path):
                raw = yaml.safe_load(path.read_text())
                config = OmegaConf.merge(
                    OmegaConf.structured(TrainConfig()),
                    OmegaConf.create(raw),
                )
                representation, allocation, filename = path.relative_to(root).parts
                size = Path(filename).stem

                self.assertEqual(
                    config.out_dir,
                    f"models/minicif_v2/{representation}/{allocation}/{size}",
                )
                self.assertEqual(config.tokenizer, "minicif_v2")
                self.assertEqual(config.batching_strategy, "record")
                self.assertTrue(config.condition_cross_attention)
                self.assertTrue(config.typed_token_heads)
                self.assertTrue(config.minicif_constrained_decoding)
                if allocation == "heavy":
                    self.assertGreater(config.pxrd_encoder_layers, 0)

                if representation in {"dense", "hybrid"}:
                    self.assertEqual(
                        config.xrd_artifact_config,
                        "configs/xrd_artifacts/full_training.yaml",
                    )
                else:
                    self.assertEqual(config.xrd_artifact_config, "")

    def test_auxiliary_v2_configs_also_use_central_output_paths(self):
        root = Path("configs/config_v2")
        auxiliary = sorted(root.glob("smoke/*.yaml")) + sorted(
            root.glob("pretrain/*.yaml")
        )

        self.assertEqual(len(auxiliary), 3)
        for path in auxiliary:
            with self.subTest(path=path):
                raw = yaml.safe_load(path.read_text())
                self.assertTrue(raw["out_dir"].startswith("models/minicif_v2/"))

    def test_xm_model_matrix_matches_standard_baselines(self):
        root = Path("configs/config_v2")
        expected = {
            root / "xm" / representation / "standard" / f"k{best_of_k}" / f"{size}.yaml"
            for representation in ("peak", "dense", "hybrid")
            for best_of_k in (2, 4, 8)
            for size in ("small", "medium", "large")
        }

        self.assertEqual(
            set(root.glob("xm/*/standard/k*/*.yaml")),
            expected,
        )
        for path in sorted(expected):
            with self.subTest(path=path):
                _, representation, allocation, k_dir, filename = path.relative_to(root).parts
                size = Path(filename).stem
                best_of_k = int(k_dir.removeprefix("k"))
                raw = yaml.safe_load(path.read_text())
                baseline = yaml.safe_load(
                    (root / representation / allocation / filename).read_text()
                )
                baseline.update({
                    "out_dir": (
                        f"models/minicif_v2/xm/{representation}/{allocation}/"
                        f"{k_dir}/{size}"
                    ),
                    "xm_best_of_k": best_of_k,
                })

                self.assertEqual(raw, baseline)


if __name__ == "__main__":
    unittest.main()
