import gzip
import importlib.util
import os
import pickle
import tempfile
import unittest

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
write_metadata = prepare_minicif_dataset.write_metadata
PrepConfig = prepare_minicif_dataset.PrepConfig


class PrepareMinicifDatasetTest(unittest.TestCase):
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
