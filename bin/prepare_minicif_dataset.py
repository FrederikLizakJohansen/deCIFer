#!/usr/bin/env python3

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from glob import glob
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from tqdm import tqdm

from decifer.minicif import MinicifConfig, MinicifTokenizer, canonicalize_cif
from decifer.utility import space_group_to_crystal_system


@dataclass
class PrepConfig:
    raw_dir: str
    out_dir: str
    num_decimal_places: int = 4
    wavelength: str = "CuKa"
    qmin: float = 0.0
    qmax: float = 10.0
    symprec: float = 0.1
    val_fraction: float = 0.075
    test_fraction: float = 0.075
    seed: int = 42
    include_occupancy_structures: bool = False
    num_workers: int = 1
    debug_max: int = 0


def process_cif(args):
    path, config_dict = args
    config = PrepConfig(**config_dict)
    name = os.path.splitext(os.path.basename(path))[0]

    structure = Structure.from_file(path)
    if not config.include_occupancy_structures:
        for site in structure:
            occupancies = list(site.species.as_dict().values())
            if any(occupancy < 1 for occupancy in occupancies):
                raise ValueError(f"{name}: occupancy below 1.0")

    cif_string = str(CifWriter(struct=structure, symprec=config.symprec))
    minicif_string = canonicalize_cif(
        cif_string,
        MinicifConfig(decimal_places=config.num_decimal_places),
    )
    tokenizer = MinicifTokenizer()
    cif_tokens = np.asarray(tokenizer.encode(tokenizer.tokenize_minicif(minicif_string)), dtype=np.int32)

    xrd_calc = XRDCalculator(wavelength=config.wavelength)
    max_q = ((4 * np.pi) / xrd_calc.wavelength) * np.sin(np.radians(90))
    if config.qmax >= max_q:
        two_theta_range = None
    else:
        tth_min = np.degrees(2 * np.arcsin((config.qmin * xrd_calc.wavelength) / (4 * np.pi)))
        tth_max = np.degrees(2 * np.arcsin((config.qmax * xrd_calc.wavelength) / (4 * np.pi)))
        two_theta_range = (tth_min, tth_max)

    pattern = xrd_calc.get_pattern(structure, two_theta_range=two_theta_range)
    theta = np.radians(pattern.x / 2)
    q_disc = (4 * np.pi * np.sin(theta) / xrd_calc.wavelength).astype(np.float32)
    iq_disc = np.asarray(pattern.y, dtype=np.float32)
    iq_disc = iq_disc / (np.max(iq_disc) + 1e-16)

    spacegroup = int(structure.get_space_group_info(symprec=config.symprec)[1])

    return {
        "cif_name": name,
        "cif_tokenized": cif_tokens,
        "minicif_string": minicif_string,
        "xrd_disc.q": q_disc,
        "xrd_disc.iq": iq_disc,
        "spacegroup": spacegroup,
        "crystal_system": space_group_to_crystal_system(spacegroup),
    }


def safe_process_cif(args):
    path, _ = args
    try:
        return process_cif(args), None
    except Exception as exc:
        return None, {"path": path, "error": str(exc)}


def write_split(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    int_vlen = h5py.vlen_dtype(np.dtype("int32"))
    float_vlen = h5py.vlen_dtype(np.dtype("float32"))

    with h5py.File(path, "w") as h5:
        h5.create_dataset("cif_name", data=[row["cif_name"] for row in rows], dtype=str_dtype)
        h5.create_dataset("minicif_string", data=[row["minicif_string"] for row in rows], dtype=str_dtype)
        h5.create_dataset("spacegroup", data=np.asarray([row["spacegroup"] for row in rows], dtype=np.int32))
        h5.create_dataset("crystal_system", data=np.asarray([row["crystal_system"] for row in rows], dtype=np.int32))

        token_ds = h5.create_dataset("cif_tokenized", shape=(len(rows),), dtype=int_vlen)
        q_ds = h5.create_dataset("xrd_disc.q", shape=(len(rows),), dtype=float_vlen)
        iq_ds = h5.create_dataset("xrd_disc.iq", shape=(len(rows),), dtype=float_vlen)

        for i, row in enumerate(rows):
            token_ds[i] = row["cif_tokenized"]
            q_ds[i] = row["xrd_disc.q"]
            iq_ds[i] = row["xrd_disc.iq"]


def split_rows(rows, val_fraction, test_fraction, seed):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n_total = len(rows)
    n_test = int(round(n_total * test_fraction))
    n_val = int(round(n_total * val_fraction))
    if n_total >= 3:
        if test_fraction > 0:
            n_test = max(1, n_test)
        if val_fraction > 0:
            n_val = max(1, n_val)
    if n_val + n_test >= n_total:
        overflow = n_val + n_test - n_total + 1
        n_val = max(0, n_val - overflow)
    n_train = n_total - n_val - n_test
    return {
        "train": rows[:n_train],
        "val": rows[n_train:n_train + n_val],
        "test": rows[n_train + n_val:],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare compact minicif HDF5 datasets directly from raw CIFs.")
    parser.add_argument("--raw-dir", required=True, help="Directory containing raw .cif files")
    parser.add_argument("--out-dir", required=True, help="Output dataset directory")
    parser.add_argument("--num-decimal-places", type=int, default=4)
    parser.add_argument("--wavelength", default="CuKa")
    parser.add_argument("--qmin", type=float, default=0.0)
    parser.add_argument("--qmax", type=float, default=10.0)
    parser.add_argument("--symprec", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.075)
    parser.add_argument("--test-fraction", type=float, default=0.075)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-occupancy-structures", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--debug-max", type=int, default=0)
    args = parser.parse_args()

    if args.num_workers == 0:
        args.num_workers = max(1, cpu_count() - 1)

    config = PrepConfig(**vars(args))
    cif_paths = sorted(glob(os.path.join(config.raw_dir, "*.cif")))
    if config.debug_max > 0:
        cif_paths = cif_paths[:config.debug_max]
    if not cif_paths:
        raise ValueError(f"no .cif files found in {config.raw_dir}")

    tasks = [(path, asdict(config)) for path in cif_paths]
    rows = []
    failures = []
    with Pool(processes=config.num_workers) as pool:
        for result, failure in tqdm(pool.imap(safe_process_cif, tasks), total=len(tasks), desc="Preparing minicif"):
            if failure is None:
                rows.append(result)
            else:
                failures.append(failure)

    splits = split_rows(rows, config.val_fraction, config.test_fraction, config.seed)
    serialized_dir = os.path.join(config.out_dir, "serialized")
    for split, split_rows_ in splits.items():
        write_split(os.path.join(serialized_dir, f"{split}.h5"), split_rows_)

    metadata = {
        "config": asdict(config),
        "n_input": len(cif_paths),
        "n_success": len(rows),
        "n_failures": len(failures),
        "split_sizes": {split: len(split_rows_) for split, split_rows_ in splits.items()},
        "tokenizer": "minicif",
        "cif_representation": "minicif",
    }
    os.makedirs(config.out_dir, exist_ok=True)
    with open(os.path.join(config.out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    if failures:
        with open(os.path.join(config.out_dir, "failures.json"), "w") as f:
            json.dump(failures, f, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
