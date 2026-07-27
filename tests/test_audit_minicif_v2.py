import importlib.util
import os
import tempfile
import unittest
from types import SimpleNamespace

import h5py
import numpy as np
import yaml
from pymatgen.core import Lattice, Structure

from bin.prepare_minicif_dataset import write_split
from bin.train import configure_tokenizer, setup_datasets, validate_training_dataset
from decifer.minicif_v2 import (
    MinicifV2Tokenizer,
    canonicalize_structure_v2,
    parse_minicif_v2,
)


MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "bin", "audit_minicif_v2.py")
spec = importlib.util.spec_from_file_location("audit_minicif_v2", MODULE_PATH)
audit_minicif_v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_minicif_v2)


class AuditMinicifV2Test(unittest.TestCase):
    def make_dataset(self, root):
        tokenizer = MinicifV2Tokenizer()
        structure = Structure.from_spacegroup(
            "Fm-3m",
            Lattice.cubic(5.64),
            ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        text = canonicalize_structure_v2(structure)
        parsed = parse_minicif_v2(text)
        tokens = np.asarray(tokenizer.encode(tokenizer.tokenize_minicif(text)), dtype=np.int32)
        row = {
            "cif_name": "nacl",
            "cif_tokenized": tokens,
            "cif_token_length": len(tokens),
            "minicif_string": text,
            "formula": "Na 1 Cl 1",
            "representation": "minicif_v2",
            "xrd_disc.q": np.asarray([1.0, 2.0], dtype=np.float32),
            "xrd_disc.iq": np.asarray([1.0, 0.5], dtype=np.float32),
            "spacegroup": parsed.space_group,
            "crystal_system": parsed.crystal_system,
        }
        serialized = os.path.join(root, "serialized")
        for split in ("train", "val", "test"):
            write_split(os.path.join(serialized, f"{split}.h5"), [row])
        config_path = os.path.join(root, "config.yaml")
        with open(config_path, "w") as handle:
            yaml.safe_dump({
                "dataset": root,
                "tokenizer": "minicif_v2",
                "batching_strategy": "record",
                "block_size": 256,
                "batch_token_budget": 512,
                "condition": True,
                "condition_cross_attention": True,
                "condition_n_tokens": 4,
                "qmin": 0.0,
                "qmax": 10.0,
            }, handle)
        return config_path

    def training_config(self, root, **overrides):
        values = {
            "dataset": root,
            "tokenizer": "minicif_v2",
            "batching_strategy": "record",
            "block_size": 256,
            "batch_token_budget": 512,
            "condition": True,
            "condition_cross_attention": True,
            "condition_n_tokens": 4,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_clean_dataset_passes_audit_and_training_guard(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = self.make_dataset(root)

            report = audit_minicif_v2.audit_dataset(config_path, max_items=1)
            configure_tokenizer("minicif_v2")
            validate_training_dataset(self.training_config(root))

        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["splits"]["train"]["n_valid_samples"], 1)

    def test_audit_rejects_wrong_representation(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = self.make_dataset(root)
            path = os.path.join(root, "serialized", "val.h5")
            with h5py.File(path, "r+") as h5:
                h5["representation"][0] = "minicif"

            report = audit_minicif_v2.audit_dataset(config_path, max_items=1)

        self.assertEqual(report["status"], "failed")
        self.assertIn("representation", report["splits"]["val"]["errors"][0]["error"])

    def test_training_guard_rejects_incompatible_block_size(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_dataset(root)
            configure_tokenizer("minicif_v2")

            with self.assertRaisesRegex(ValueError, "no records that fit block_size"):
                validate_training_dataset(self.training_config(root, block_size=8))

    def test_overlength_records_are_reported_and_excluded(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_dataset(root)
            train_path = os.path.join(root, "serialized", "train.h5")
            with h5py.File(train_path, "r") as h5:
                row = {
                    "cif_name": h5["cif_name"].asstr()[0],
                    "cif_tokenized": np.asarray(h5["cif_tokenized"][0], dtype=np.int32),
                    "cif_token_length": int(h5["cif_token_length"][0]),
                    "minicif_string": h5["minicif_string"].asstr()[0],
                    "formula": h5["formula"].asstr()[0],
                    "representation": h5["representation"].asstr()[0],
                    "xrd_backend": h5["xrd_backend"].asstr()[0],
                    "xrd_disc.q": np.asarray(h5["xrd_disc.q"][0], dtype=np.float32),
                    "xrd_disc.iq": np.asarray(h5["xrd_disc.iq"][0], dtype=np.float32),
                    "spacegroup": int(h5["spacegroup"][0]),
                    "crystal_system": int(h5["crystal_system"][0]),
                }
            overlength_row = dict(row)
            overlength_row["cif_name"] = "overlength"
            overlength_row["cif_tokenized"] = np.resize(row["cif_tokenized"], 300)
            overlength_row["cif_token_length"] = 300
            write_split(train_path, [row, overlength_row])
            config = self.training_config(
                root,
                block_size=256,
                device="cpu",
                sampling_strategy="random",
                batch_size=1,
                num_workers_dataloader=0,
                seed=42,
                length_bucket_size=16,
            )

            stats = validate_training_dataset(config)
            dataloaders = setup_datasets(config)

            self.assertEqual(stats["train"]["n_excluded"], 1)
            self.assertEqual(stats["val"]["n_excluded"], 0)
            self.assertEqual(len(dataloaders["train"].dataset), 1)
            for dataloader in dataloaders.values():
                dataloader.dataset.close()


if __name__ == "__main__":
    unittest.main()
