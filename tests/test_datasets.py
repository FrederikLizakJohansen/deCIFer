import os

import pytest

from decifer.datasets import (
    load_decifer_dataset,
    resolve_dataset_file,
    resolve_dataset_splits,
)


def test_resolve_dataset_file_accepts_explicit_h5_path(tmp_path):
    dataset_file = tmp_path / "test.h5"
    dataset_file.write_text("")

    resolved = resolve_dataset_file(str(dataset_file))

    assert resolved == str(dataset_file.resolve())


def test_resolve_dataset_file_accepts_dataset_root_with_split(tmp_path):
    serialized_dir = tmp_path / "dataset_root" / "serialized"
    serialized_dir.mkdir(parents=True)
    test_file = serialized_dir / "test.h5"
    test_file.write_text("")

    resolved = resolve_dataset_file(str(tmp_path / "dataset_root"), split="test")

    assert resolved == str(test_file.resolve())


def test_resolve_dataset_splits_returns_all_standard_split_paths(tmp_path):
    serialized_dir = tmp_path / "dataset_root" / "serialized"
    serialized_dir.mkdir(parents=True)
    for split in ["train", "val", "test"]:
        (serialized_dir / f"{split}.h5").write_text("")

    resolved = resolve_dataset_splits(str(tmp_path / "dataset_root"))

    assert set(resolved) == {"train", "val", "test"}
    assert resolved["train"].endswith(os.path.join("serialized", "train.h5"))
    assert resolved["val"].endswith(os.path.join("serialized", "val.h5"))
    assert resolved["test"].endswith(os.path.join("serialized", "test.h5"))


def test_load_decifer_dataset_resolves_root_before_construction(tmp_path):
    serialized_dir = tmp_path / "dataset_root" / "serialized"
    serialized_dir.mkdir(parents=True)
    val_file = serialized_dir / "val.h5"
    val_file.write_text("")

    calls = {}

    class FakeDataset:
        def __init__(self, path, keys):
            calls["path"] = path
            calls["keys"] = keys

    dataset = load_decifer_dataset(
        str(tmp_path / "dataset_root"),
        ["cif_name"],
        split="val",
        dataset_cls=FakeDataset,
    )

    assert isinstance(dataset, FakeDataset)
    assert calls["path"] == str(val_file.resolve())
    assert calls["keys"] == ["cif_name"]


def test_resolve_dataset_file_requires_split_for_directory(tmp_path):
    dataset_root = tmp_path / "dataset_root"
    dataset_root.mkdir()

    with pytest.raises(ValueError, match="dataset split is required"):
        resolve_dataset_file(str(dataset_root))
