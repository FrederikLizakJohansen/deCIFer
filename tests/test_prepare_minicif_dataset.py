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
save_checkpoint = prepare_minicif_dataset.save_checkpoint
select_inputs = prepare_minicif_dataset.select_inputs


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


if __name__ == "__main__":
    unittest.main()
