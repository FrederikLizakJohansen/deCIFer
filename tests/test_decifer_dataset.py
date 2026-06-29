import os
import tempfile
import unittest
from types import SimpleNamespace

import h5py
import numpy as np
from torch.utils.data import WeightedRandomSampler

from decifer.decifer_dataset import DeciferDataset


class DeciferDatasetTest(unittest.TestCase):
    def test_scalar_numeric_fields_are_loaded_as_tensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.h5")
            with h5py.File(path, "w") as h5:
                h5.create_dataset("spacegroup", data=np.asarray([221], dtype=np.int32))

            dataset = DeciferDataset(path, ["spacegroup"])
            item = dataset[0]

            self.assertEqual(item["spacegroup"].item(), 221)

    def test_lazy_open_reopens_file_on_access(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.h5")
            with h5py.File(path, "w") as h5:
                h5.create_dataset("spacegroup", data=np.asarray([221, 225], dtype=np.int32))

            dataset = DeciferDataset(path, ["spacegroup"], lazy_open=True)

            self.assertIsNone(dataset.h5_file)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[1]["spacegroup"].item(), 225)
            self.assertIsNotNone(dataset.h5_file)

    def test_crystal_system_balanced_training_sampler_is_weighted(self):
        from bin.train import setup_datasets

        with tempfile.TemporaryDirectory() as tmpdir:
            serialized = os.path.join(tmpdir, "serialized")
            os.makedirs(serialized)
            for split in ["train", "val", "test"]:
                path = os.path.join(serialized, f"{split}.h5")
                with h5py.File(path, "w") as h5:
                    int_vlen = h5py.vlen_dtype(np.dtype("int32"))
                    float_vlen = h5py.vlen_dtype(np.dtype("float32"))
                    tokens = h5.create_dataset("cif_tokenized", shape=(4,), dtype=int_vlen)
                    q = h5.create_dataset("xrd_disc.q", shape=(4,), dtype=float_vlen)
                    iq = h5.create_dataset("xrd_disc.iq", shape=(4,), dtype=float_vlen)
                    h5.create_dataset("crystal_system", data=np.asarray([1, 1, 1, 7], dtype=np.int32))
                    for i in range(4):
                        tokens[i] = np.asarray([0, 1], dtype=np.int32)
                        q[i] = np.asarray([1.0], dtype=np.float32)
                        iq[i] = np.asarray([1.0], dtype=np.float32)

            config = SimpleNamespace(
                dataset=tmpdir,
                sampling_strategy="crystal_system_balanced",
                device="cpu",
                batch_size=2,
                num_workers_dataloader=0,
                seed=42,
            )

            dataloaders = setup_datasets(config)

        self.assertIsInstance(dataloaders["train"].batch_sampler.sampler, WeightedRandomSampler)


if __name__ == "__main__":
    unittest.main()
