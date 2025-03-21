#!/usr/bin/env python3

import os
import argparse
import h5py
import numpy as np

def combine_h5_files(h5_files, output_h5):
    if not h5_files:
        print(f"No files to combine for {output_h5}. Skipping.")
        return

    # Step A: Identify the union of all dataset keys
    all_keys = set()
    for path in h5_files:
        with h5py.File(path, "r") as f_in:
            all_keys.update(f_in.keys())

    # Step B: Read data in memory (still chunkable, but for brevity we do full read)
    #         If your data is huge, see the chunk-based approach below.
    data_map = {k: [] for k in all_keys}
    for path in h5_files:
        with h5py.File(path, "r") as f_in:
            file_keys = f_in.keys()
            for k in all_keys:
                # If the file doesn’t have this dataset, skip
                if k not in file_keys:
                    continue
                arr = f_in[k][()]
                data_map[k].append(arr)

    # Step C: Create the output file and write each dataset
    with h5py.File(output_h5, "w") as f_out:
        for k, list_of_arrays in data_map.items():
            if not list_of_arrays:
                continue

            # 1) Concatenate them along the first dimension
            total_length = sum(a.shape[0] for a in list_of_arrays)

            # 2) Inspect the *first element of the first array* 
            #    to guess how to store things
            sample_item = list_of_arrays[0][0]  # e.g., could be np.ndarray or bytes or str

            # Decide if numeric vlen or string vlen
            # (You can extend logic for arrays-of-int, arrays-of-float, etc.)
            if isinstance(sample_item, np.ndarray):
                # Suppose it’s numeric (float or int) arrays
                # Make a vlen dtype of the same type
                if sample_item.dtype.kind in ("i","f"):
                    vlen_dtype = h5py.vlen_dtype(sample_item.dtype)
                else:
                    # fallback - store as vlen string
                    vlen_dtype = h5py.vlen_dtype(str)
            elif isinstance(sample_item, (str, bytes)):
                # textual data
                vlen_dtype = h5py.vlen_dtype(str)
            else:
                # fallback: store as vlen string (or raise an error)
                vlen_dtype = h5py.vlen_dtype(str)

            dset = f_out.create_dataset(
                k,
                shape=(total_length,),
                dtype=vlen_dtype
            )

            # 3) Chunked writing (to avoid big concatenations):
            idx = 0
            for arr in list_of_arrays:
                n = arr.shape[0]
                for i in range(n):
                    dset[idx] = arr[i]  # store the sub-array (or string) as vlen
                    idx += 1

    print(f"Finished creating {output_h5}.\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", nargs='+', required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs='+', default=["train", "val", "test"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split in args.splits:
        h5_files = []
        for d in args.input_dirs:
            candidate = os.path.join(d, f"{split}.h5")
            if os.path.isfile(candidate):
                h5_files.append(candidate)
        if not h5_files:
            continue

        output_h5 = os.path.join(args.output_dir, f"{split}.h5")
        combine_h5_files(h5_files, output_h5)

    print("All done!")

if __name__ == "__main__":
    main()
