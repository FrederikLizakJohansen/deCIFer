import gzip
import importlib.util
import os
import pickle
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "bin", "prepare_minicif_dataset.py")
spec = importlib.util.spec_from_file_location("prepare_minicif_dataset", MODULE_PATH)
prepare_minicif_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_minicif_dataset)
load_inputs = prepare_minicif_dataset.load_inputs
load_checkpoint = prepare_minicif_dataset.load_checkpoint
load_shard_checkpoints = prepare_minicif_dataset.load_shard_checkpoints
runtime_exceeded = prepare_minicif_dataset.runtime_exceeded
save_checkpoint = prepare_minicif_dataset.save_checkpoint
select_inputs = prepare_minicif_dataset.select_inputs
shard_checkpoint_path = prepare_minicif_dataset.shard_checkpoint_path
shard_inputs = prepare_minicif_dataset.shard_inputs
split_rows = prepare_minicif_dataset.split_rows
filter_rows_by_token_length = prepare_minicif_dataset.filter_rows_by_token_length
write_metadata = prepare_minicif_dataset.write_metadata
write_split = prepare_minicif_dataset.write_split
process_cif = prepare_minicif_dataset.process_cif
validate_checkpoint_representation = prepare_minicif_dataset.validate_checkpoint_representation
validate_checkpoint_xrd_backend = prepare_minicif_dataset.validate_checkpoint_xrd_backend
calculate_xrd_pattern = prepare_minicif_dataset.calculate_xrd_pattern
PrepConfig = prepare_minicif_dataset.PrepConfig


class PrepareMinicifDatasetTest(unittest.TestCase):
    def test_overlength_target_is_rejected_before_diffraction(self):
        structure = Structure(
            Lattice.cubic(4.0),
            ["Na"],
            [[0.0, 0.0, 0.0]],
        )
        config = PrepConfig(
            raw_dir="raw",
            out_dir="out",
            representation="minicif_v2",
            max_token_length=1,
        )

        with patch.object(prepare_minicif_dataset, "calculate_xrd_pattern") as calculate:
            with self.assertRaisesRegex(ValueError, "exceeds --max-token-length 1"):
                process_cif((("nacl", str(CifWriter(structure))), vars(config)))

        calculate.assert_not_called()

    def test_filter_rows_by_token_length_handles_resumed_checkpoints(self):
        rows = [
            {"cif_name": "short", "cif_tokenized": [1, 2], "cif_token_length": 2},
            {"cif_name": "long", "cif_tokenized": [1, 2, 3], "cif_token_length": 3},
        ]

        kept, excluded = filter_rows_by_token_length(rows, max_token_length=2)

        self.assertEqual([row["cif_name"] for row in kept], ["short"])
        self.assertEqual(excluded[0]["source"], "long")
        self.assertIn("token length 3", excluded[0]["error"])

    def test_v2_processing_writes_formula_representation_and_token_length(self):
        structure = Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        )
        config = PrepConfig(raw_dir="raw", out_dir="out", representation="minicif_v2")

        row = process_cif((("nacl", str(CifWriter(structure))), vars(config)))

        self.assertEqual(row["representation"], "minicif_v2")
        self.assertEqual(row["formula"], "Na 1 Cl 1")
        self.assertEqual(row["cif_token_length"], len(row["cif_tokenized"]))
        self.assertTrue(row["minicif_string"].startswith("<mcif2>"))
        self.assertIn(row["xrd_backend"], {"braggcalculator", "pymatgen"})

    def test_braggcalculator_matches_pymatgen_sparse_pattern(self):
        try:
            __import__("braggcalculator")
        except ImportError:
            self.skipTest("braggcalculator is not installed")
        structure = Structure.from_spacegroup(
            "Pm-3m",
            Lattice.cubic(3.905),
            ["Sr", "Ti", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]],
        )
        pymatgen_config = PrepConfig(
            raw_dir="raw", out_dir="out", xrd_backend="pymatgen", qmax=8.0
        )
        bragg_config = PrepConfig(
            raw_dir="raw", out_dir="out", xrd_backend="braggcalculator", qmax=8.0
        )

        pymatgen_q, pymatgen_iq, _ = calculate_xrd_pattern(structure, pymatgen_config)
        bragg_q, bragg_iq, _ = calculate_xrd_pattern(structure, bragg_config)

        np.testing.assert_allclose(bragg_q, pymatgen_q, rtol=0, atol=1e-6)
        np.testing.assert_allclose(bragg_iq, pymatgen_iq, rtol=0, atol=1e-6)

    def test_v2_split_contains_batching_metadata_fields(self):
        row = {
            "cif_name": "a",
            "minicif_string": "<mcif2> Na formula Na 1",
            "formula": "Na 1",
            "representation": "minicif_v2",
            "xrd_backend": "braggcalculator",
            "cif_tokenized": [1, 2, 3],
            "cif_token_length": 3,
            "spacegroup": 1,
            "crystal_system": 1,
            "xrd_disc.q": [1.0],
            "xrd_disc.iq": [1.0],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "train.h5")
            write_split(path, [row])
            with h5py.File(path, "r") as h5:
                self.assertEqual(int(h5["cif_token_length"][0]), 3)
                self.assertEqual(h5["formula"].asstr()[0], "Na 1")
                self.assertEqual(h5["representation"].asstr()[0], "minicif_v2")
                self.assertEqual(h5["xrd_backend"].asstr()[0], "braggcalculator")

    def test_checkpoint_representation_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "another representation"):
            validate_checkpoint_representation(
                {"a": {"representation": "minicif"}}, "minicif_v2"
            )

    def test_checkpoint_xrd_backend_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "another XRD backend"):
            validate_checkpoint_xrd_backend(
                {"a": {"xrd_backend": "pymatgen"}}, "braggcalculator"
            )

    def test_load_inputs_from_gzip_tuple_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = os.path.join(tmpdir, "raw.pkl.gz")
            with gzip.open(bundle_path, "wb") as f:
                pickle.dump([("sample_a", "data_a"), ("sample_b.cif", "data_b")], f)

            inputs = load_inputs(tmpdir, raw_from_gzip=True)

        self.assertEqual(inputs, [("sample_a", "data_a"), ("sample_b.cif", "data_b")])

    def test_select_inputs_first_and_random(self):
        inputs = list(range(10))

        self.assertEqual(select_inputs(inputs, 3, "first", seed=42), [0, 1, 2])
        self.assertEqual(select_inputs(inputs, 0, "first", seed=42), inputs)
        self.assertEqual(len(select_inputs(inputs, 4, "random", seed=42)), 4)
        self.assertEqual(select_inputs(inputs, 4, "random", seed=42), select_inputs(inputs, 4, "random", seed=42))

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "prep_checkpoint.pkl.gz")
            rows = {"a": {"cif_name": "a"}}
            failures = {"b": {"source": "b", "error": "bad"}}

            save_checkpoint(checkpoint_path, rows, failures)
            loaded_rows, loaded_failures = load_checkpoint(checkpoint_path)

        self.assertEqual(loaded_rows, rows)
        self.assertEqual(loaded_failures, failures)

    def test_checkpoint_round_trip_without_directory_component(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                rows = {"a": {"cif_name": "a"}}
                failures = {}

                save_checkpoint("prep_checkpoint.pkl.gz", rows, failures)
                loaded_rows, loaded_failures = load_checkpoint("prep_checkpoint.pkl.gz")
            finally:
                os.chdir(old_cwd)

        self.assertEqual(loaded_rows, rows)
        self.assertEqual(loaded_failures, failures)

    def test_shard_inputs_use_index_modulo(self):
        inputs = list(range(10))

        self.assertEqual(shard_inputs(inputs, shard_index=0, num_shards=3), [0, 3, 6, 9])
        self.assertEqual(shard_inputs(inputs, shard_index=1, num_shards=3), [1, 4, 7])
        self.assertEqual(shard_inputs(inputs, shard_index=2, num_shards=3), [2, 5, 8])

    def test_shard_checkpoint_paths_keep_pkl_gz_suffix(self):
        path = shard_checkpoint_path("out/prep_checkpoint.pkl.gz", shard_index=2, num_shards=16)

        self.assertEqual(path, "out/prep_checkpoint_shard_00002_of_00016.pkl.gz")

    def test_load_shard_checkpoints_merges_rows_and_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "prep_checkpoint.pkl.gz")
            save_checkpoint(
                shard_checkpoint_path(base_path, 0, 2),
                {"a": {"cif_name": "a"}},
                {},
            )
            save_checkpoint(
                shard_checkpoint_path(base_path, 1, 2),
                {"b": {"cif_name": "b"}},
                {"c": {"source": "c", "error": "bad"}},
            )

            rows, failures = load_shard_checkpoints(base_path, 2)

        self.assertEqual(set(rows), {"a", "b"})
        self.assertEqual(set(failures), {"c"})

    def test_runtime_exceeded_respects_disabled_zero(self):
        self.assertFalse(runtime_exceeded(0.0, 0))
        self.assertTrue(runtime_exceeded(0.0, 1))

    def test_incomplete_metadata_records_checkpoint_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PrepConfig(raw_dir="raw", out_dir=tmpdir, checkpoint_path=os.path.join(tmpdir, "ckpt.pkl.gz"))

            metadata = write_metadata(
                config=config,
                inputs=["a", "b"],
                rows=[{"cif_name": "a"}],
                failures=[],
                splits={},
                pending_inputs=["b"],
                n_processed_this_run=1,
                complete=False,
                stop_reason="max_runtime_seconds",
            )

            with open(os.path.join(tmpdir, "metadata.json")) as f:
                loaded = __import__("json").load(f)

        self.assertFalse(metadata["complete"])
        self.assertEqual(loaded["stop_reason"], "max_runtime_seconds")

    def test_split_rows_stratifies_by_crystal_system(self):
        rows = [
            {"cif_name": f"cs{crystal_system}_{i}", "crystal_system": crystal_system}
            for crystal_system in [1, 2, 7]
            for i in range(20)
        ]

        splits = split_rows(rows, val_fraction=0.1, test_fraction=0.1, seed=42, stratify_on="crystal_system")

        for split in ["train", "val", "test"]:
            crystal_systems = {row["crystal_system"] for row in splits[split]}
            self.assertEqual(crystal_systems, {1, 2, 7})


if __name__ == "__main__":
    unittest.main()
