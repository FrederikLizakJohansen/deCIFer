#!/usr/bin/env python3

import argparse
import gzip
import json
import os
import pickle
import random
import sys
from dataclasses import asdict, dataclass
from glob import glob
from multiprocessing import Pool, cpu_count

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

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
    raw_from_gzip: bool = False
    max_samples: int = 0
    sample_strategy: str = "first"
    checkpoint_path: str = ""
    checkpoint_interval: int = 1000
    no_resume: bool = False
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
    obj, config_dict = args
    config = PrepConfig(**config_dict)
    if isinstance(obj, tuple):
        name, cif_string_raw = obj
        structure = Structure.from_str(cif_string_raw, fmt="cif")
        name = os.path.splitext(str(name))[0]
    else:
        name = os.path.splitext(os.path.basename(obj))[0]
        structure = Structure.from_file(obj)

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
    obj, _ = args
    key = source_id(obj)
    try:
        return key, process_cif(args), None
    except Exception as exc:
        return key, None, {"source": key, "error": str(exc)}


def load_inputs(raw_dir, raw_from_gzip):
    if raw_from_gzip:
        bundle_paths = sorted(glob(os.path.join(raw_dir, "*.pkl.gz")))
        if not bundle_paths:
            raise ValueError(f"no .pkl.gz files found in {raw_dir}")
        inputs = []
        for bundle_path in bundle_paths:
            with gzip.open(bundle_path, "rb") as f:
                bundle = pickle.load(f)
            for item in bundle:
                if isinstance(item, tuple) and len(item) >= 2:
                    inputs.append((item[0], item[1]))
                elif isinstance(item, dict) and "cif_name" in item and "cif_string" in item:
                    inputs.append((item["cif_name"], item["cif_string"]))
                else:
                    raise ValueError(f"unsupported raw gzip entry in {bundle_path}: {type(item)}")
        return inputs

    cif_paths = sorted(glob(os.path.join(raw_dir, "*.cif")))
    if not cif_paths:
        raise ValueError(f"no .cif files found in {raw_dir}")
    return cif_paths


def source_id(obj):
    if isinstance(obj, tuple):
        return str(obj[0])
    return str(obj)


def select_inputs(inputs, max_samples, sample_strategy, seed):
    inputs = list(inputs)
    if max_samples <= 0 or max_samples >= len(inputs):
        return inputs
    if sample_strategy == "first":
        return inputs[:max_samples]
    if sample_strategy == "random":
        rng = random.Random(seed)
        return rng.sample(inputs, max_samples)
    raise ValueError(f"unknown sample_strategy: {sample_strategy}")


def load_checkpoint(path):
    if not path or not os.path.exists(path):
        return {}, {}
    with gzip.open(path, "rb") as f:
        state = pickle.load(f)
    rows = {row["source_id"]: row["data"] for row in state.get("rows", [])}
    failures = {failure["source_id"]: failure["data"] for failure in state.get("failures", [])}
    return rows, failures


def save_checkpoint(path, rows_by_source, failures_by_source):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "rows": [{"source_id": key, "data": value} for key, value in rows_by_source.items()],
        "failures": [{"source_id": key, "data": value} for key, value in failures_by_source.items()],
    }
    tmp_path = path + ".tmp"
    with gzip.open(tmp_path, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp_path, path)


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
    parser.add_argument("--raw-dir", required=True, help="Directory containing raw .cif files or .pkl.gz bundles")
    parser.add_argument("--out-dir", required=True, help="Output dataset directory")
    parser.add_argument("--raw-from-gzip", action="store_true", help="Read raw CIF strings from .pkl.gz bundle(s)")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit the number of raw inputs after deterministic selection")
    parser.add_argument("--sample-strategy", choices=["first", "random"], default="first", help="How to select --max-samples inputs")
    parser.add_argument("--checkpoint-path", default="", help="Path to resumable prep checkpoint; defaults to OUT_DIR/prep_checkpoint.pkl.gz")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Save prep checkpoint every N processed inputs")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any existing prep checkpoint")
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
    inputs = load_inputs(config.raw_dir, config.raw_from_gzip)
    inputs = select_inputs(inputs, config.max_samples, config.sample_strategy, config.seed)
    if config.debug_max > 0:
        inputs = inputs[:config.debug_max]

    if not config.checkpoint_path:
        config.checkpoint_path = os.path.join(config.out_dir, "prep_checkpoint.pkl.gz")

    rows_by_source = {}
    failures_by_source = {}
    if not config.no_resume:
        rows_by_source, failures_by_source = load_checkpoint(config.checkpoint_path)

    pending_inputs = [
        obj for obj in inputs
        if source_id(obj) not in rows_by_source and source_id(obj) not in failures_by_source
    ]
    tasks = [(obj, asdict(config)) for obj in pending_inputs]
    with Pool(processes=config.num_workers) as pool:
        iterator = pool.imap(safe_process_cif, tasks)
        for n_processed, (key, result, failure) in enumerate(tqdm(iterator, total=len(tasks), desc="Preparing minicif"), start=1):
            if failure is None:
                rows_by_source[key] = result
            else:
                failures_by_source[key] = failure
            if config.checkpoint_interval > 0 and n_processed % config.checkpoint_interval == 0:
                save_checkpoint(config.checkpoint_path, rows_by_source, failures_by_source)

    save_checkpoint(config.checkpoint_path, rows_by_source, failures_by_source)

    rows = [
        rows_by_source[source_id(obj)]
        for obj in inputs
        if source_id(obj) in rows_by_source
    ]
    failures = list(failures_by_source.values())

    splits = split_rows(rows, config.val_fraction, config.test_fraction, config.seed)
    serialized_dir = os.path.join(config.out_dir, "serialized")
    for split, split_rows_ in splits.items():
        write_split(os.path.join(serialized_dir, f"{split}.h5"), split_rows_)

    metadata = {
        "config": asdict(config),
        "n_input": len(inputs),
        "n_pending_at_start": len(pending_inputs),
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
