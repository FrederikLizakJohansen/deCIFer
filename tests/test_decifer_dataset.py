import sys
import types

import numpy as np
import torch

from conftest import REPO_ROOT, load_module_from_path


class FakeDataset:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, idx):
        return self.values[idx]

    def __len__(self):
        return len(self.values)


class FakeFile(dict):
    pass


def install_fake_h5py(monkeypatch, fake_file):
    fake_h5py = types.ModuleType("h5py")
    fake_h5py.Dataset = FakeDataset
    fake_h5py.File = lambda path, mode: fake_file
    monkeypatch.setitem(sys.modules, "h5py", fake_h5py)


def test_dataset_key_mapping_and_type_conversion(monkeypatch):
    fake_file = FakeFile(
        {
            "cif_tokenized": FakeDataset([np.array([1, 2, 3], dtype=np.int64)]),
            "xrd_disc.q": FakeDataset([np.array([0.1, 0.2], dtype=np.float32)]),
            "xrd_disc.iq": FakeDataset([np.array([1.0, 0.5], dtype=np.float32)]),
            "cif_name": FakeDataset([b"sample_0.cif"]),
        }
    )
    install_fake_h5py(monkeypatch, fake_file)

    module = load_module_from_path(
        REPO_ROOT / "decifer" / "decifer_dataset.py",
        "test_decifer_dataset_module",
    )

    dataset = module.DeciferDataset(
        "ignored.h5", ["cif_tokens", "xrd.q", "xrd.iq", "cif_name"]
    )
    item = dataset[0]

    assert torch.equal(item["cif_tokens"], torch.tensor([1, 2, 3], dtype=torch.long))
    assert torch.equal(item["xrd.q"], torch.tensor([0.1, 0.2], dtype=torch.float32))
    assert torch.equal(item["xrd.iq"], torch.tensor([1.0, 0.5], dtype=torch.float32))
    assert item["cif_name"] == "sample_0.cif"
