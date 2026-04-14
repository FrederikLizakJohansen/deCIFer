#!/usr/bin/env python3

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional


VALID_DATASET_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetSpec:
    source_path: str
    split: Optional[str] = None


def is_hdf5_path(path: str) -> bool:
    return path.endswith(".h5")


def _normalize_split(split: Optional[str]) -> Optional[str]:
    if split is None:
        return None
    split = split.lower()
    if split not in VALID_DATASET_SPLITS:
        valid = ", ".join(VALID_DATASET_SPLITS)
        raise ValueError(f"Unknown dataset split '{split}'. Expected one of: {valid}")
    return split


def _candidate_dataset_paths(root_path: str, split: str) -> Iterable[str]:
    yield os.path.join(root_path, "serialized", f"{split}.h5")
    yield os.path.join(root_path, f"{split}.h5")


def resolve_dataset_file(source_path: str, split: Optional[str] = None) -> str:
    source_path = os.path.abspath(source_path)
    split = _normalize_split(split)

    if is_hdf5_path(source_path):
        return source_path

    if not os.path.isdir(source_path):
        raise FileNotFoundError(f"Dataset path does not exist: {source_path}")

    if split is None:
        raise ValueError(
            "A dataset split is required when the dataset path points to a directory"
        )

    for candidate in _candidate_dataset_paths(source_path, split):
        if is_hdf5_path(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not resolve split '{split}' under dataset directory: {source_path}"
    )


def resolve_dataset_splits(source_path: str) -> Dict[str, str]:
    return {
        split: resolve_dataset_file(source_path, split=split)
        for split in VALID_DATASET_SPLITS
    }


def load_decifer_dataset(
    source_path: str,
    data_keys,
    split: Optional[str] = None,
    dataset_cls=None,
):
    if dataset_cls is None:
        from decifer.decifer_dataset import DeciferDataset

        dataset_cls = DeciferDataset
    dataset_file = resolve_dataset_file(source_path, split=split)
    return dataset_cls(dataset_file, data_keys)
