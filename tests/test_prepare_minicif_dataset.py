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


class PrepareMinicifDatasetTest(unittest.TestCase):
    def test_load_inputs_from_gzip_tuple_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = os.path.join(tmpdir, "raw.pkl.gz")
            with gzip.open(bundle_path, "wb") as f:
                pickle.dump([("sample_a", "data_a"), ("sample_b.cif", "data_b")], f)

            inputs = load_inputs(tmpdir, raw_from_gzip=True)

        self.assertEqual(inputs, [("sample_a", "data_a"), ("sample_b.cif", "data_b")])


if __name__ == "__main__":
    unittest.main()
