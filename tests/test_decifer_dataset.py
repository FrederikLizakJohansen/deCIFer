import os
import tempfile
import unittest

import h5py
import numpy as np

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


if __name__ == "__main__":
    unittest.main()
