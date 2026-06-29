#!/usr/bin/env python3

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class DeciferDataset(Dataset):

    KEY_MAPPINGS = {
        'cif_tokens': 'cif_tokenized',
        'xrd.q': 'xrd_disc.q',
        'xrd.iq': 'xrd_disc.iq',
    }

    def __init__(self, h5_path, data_keys, lazy_open=False):
        # Key mappings for backward compatibility
        self.h5_path = h5_path
        self.data_keys = data_keys
        self.lazy_open = lazy_open
        self.h5_file = None
        self.data = {}
        self.dataset_length = 0

        self._open_file()
        if self.lazy_open:
            self.close()

    def _open_file(self):
        if self.h5_file is not None:
            return

        # Ensure that data_keys only contain datasets
        self.h5_file = h5py.File(self.h5_path, 'r')
        self.data = {}
        for key in self.data_keys:
            # Resolve mapped key or fallback to original
            mapped_key = self.KEY_MAPPINGS.get(key)
            if mapped_key and mapped_key in self.h5_file:
                item = self.h5_file[mapped_key]
            elif key in self.h5_file:
                item = self.h5_file[key]
            else:
                raise KeyError(f"Neither '{key}' nor its mapped key exists in the HDF5 file")

            # Validate type
            if isinstance(item, h5py.Dataset):
                self.data[key] = item
            else:
                raise TypeError(f"The key '{key}' does not correspond to an h5py.Dataset.")

        self.dataset_length = len(next(iter(self.data.values())))

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
        self.h5_file = None
        self.data = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["h5_file"] = None
        state["data"] = {}
        return state

    def __del__(self):
        self.close()

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        self._open_file()
        data = {}
        for key in self.data_keys:
            sequence = self.data[key][idx]

            # Handle numeric data (np.ndarray)
            if isinstance(sequence, np.ndarray):
                dtype = torch.float32 if 'float' in str(sequence.dtype) else torch.long
                sequence = torch.tensor(sequence, dtype=dtype)
            elif isinstance(sequence, np.generic):
                dtype = torch.float32 if 'float' in str(sequence.dtype) else torch.long
                sequence = torch.tensor(sequence.item(), dtype=dtype)
            elif isinstance(sequence, (bytes, str)):
                sequence = sequence.decode('utf-8') if isinstance(sequence, bytes) else sequence
            else:
                raise TypeError(f"Unsupported sequence type {type(sequence)}")

            data[key] = sequence

        return data
