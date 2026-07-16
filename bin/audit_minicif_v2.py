#!/usr/bin/env python3

import argparse
import json
import os
import random
import sys

import h5py
import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from decifer.minicif_v2 import (
    END_TOKEN,
    START_TOKEN,
    MinicifV2Tokenizer,
    minicif_v2_to_structure,
    parse_minicif_v2,
)


REQUIRED_DATASETS = {
    "cif_tokenized",
    "cif_token_length",
    "minicif_string",
    "formula",
    "representation",
    "xrd_disc.q",
    "xrd_disc.iq",
    "spacegroup",
    "crystal_system",
}


def decode_string(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_config(path):
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    if config.get("tokenizer") != "minicif_v2":
        raise ValueError(f"{path} must set tokenizer: minicif_v2")
    if config.get("batching_strategy") != "record":
        raise ValueError(f"{path} must set batching_strategy: record")
    return config


def split_path(dataset_dir, split):
    candidates = [
        os.path.join(dataset_dir, "serialized", f"{split}.h5"),
        os.path.join(dataset_dir, f"{split}.h5"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"could not find {split}.h5 under {dataset_dir}")


def sample_indices(n_records, max_items, seed):
    if max_items <= 0 or max_items >= n_records:
        return list(range(n_records))
    return sorted(random.Random(seed).sample(range(n_records), max_items))


def add_error(report, index, message):
    report["n_errors"] += 1
    if len(report["errors"]) < 50:
        report["errors"].append({"index": int(index), "error": str(message)})


def validate_sample(h5, index, tokenizer, config, report):
    try:
        representation = decode_string(h5["representation"][index])
        if representation != "minicif_v2":
            raise ValueError(f"representation is {representation!r}")
        xrd_backend = (
            decode_string(h5["xrd_backend"][index])
            if "xrd_backend" in h5
            else "pymatgen"
        )
        if xrd_backend != report["xrd_backend"]:
            raise ValueError(
                f"XRD backend {xrd_backend!r} differs from {report['xrd_backend']!r}"
            )

        tokens = np.asarray(h5["cif_tokenized"][index], dtype=np.int64)
        stored_length = int(h5["cif_token_length"][index])
        if len(tokens) != stored_length:
            raise ValueError(f"stored token length {stored_length} != actual {len(tokens)}")
        if len(tokens) == 0 or tokens.min() < 0 or tokens.max() >= tokenizer.vocab_size:
            raise ValueError("token ids are empty or outside the minicif_v2 vocabulary")

        text = decode_string(h5["minicif_string"][index])
        if tokenizer.decode(tokens.tolist()) != text:
            raise ValueError("cif_tokenized does not decode to minicif_string")
        if not text.startswith(START_TOKEN) or not text.endswith(END_TOKEN):
            raise ValueError("minicif_string has invalid boundary tokens")

        parsed = parse_minicif_v2(text)
        structure = minicif_v2_to_structure(text)
        if parsed.space_group != int(h5["spacegroup"][index]):
            raise ValueError("spacegroup metadata does not match minicif_string")
        if parsed.crystal_system != int(h5["crystal_system"][index]):
            raise ValueError("crystal_system metadata does not match minicif_string")
        expected_formula = " ".join(
            f"{element} {parsed.formula[element]}" for element in parsed.elements
        )
        if decode_string(h5["formula"][index]) != expected_formula:
            raise ValueError("formula metadata does not match minicif_string")
        if structure.composition.num_atoms <= 0:
            raise ValueError("expanded structure contains no atoms")

        q = np.asarray(h5["xrd_disc.q"][index], dtype=np.float64)
        iq = np.asarray(h5["xrd_disc.iq"][index], dtype=np.float64)
        if q.size == 0 or q.shape != iq.shape:
            raise ValueError("PXRD q/intensity arrays are empty or have different lengths")
        if not np.isfinite(q).all() or not np.isfinite(iq).all():
            raise ValueError("PXRD arrays contain non-finite values")
        if (iq < 0).any():
            raise ValueError("PXRD intensities contain negative values")
        qmin = float(config.get("qmin", 0.0))
        qmax = float(config.get("qmax", 10.0))
        if q.min() < qmin - 1e-6 or q.max() > qmax + 1e-6:
            raise ValueError(f"PXRD q values fall outside configured [{qmin}, {qmax}]")
        report["n_valid_samples"] += 1
    except Exception as exc:
        add_error(report, index, exc)


def audit_split(path, config, max_items=100, seed=1337):
    tokenizer = MinicifV2Tokenizer()
    report = {
        "path": os.path.abspath(path),
        "n_records": 0,
        "n_sampled": 0,
        "n_valid_samples": 0,
        "n_errors": 0,
        "errors": [],
    }
    with h5py.File(path, "r") as h5:
        missing = sorted(REQUIRED_DATASETS - set(h5.keys()))
        if missing:
            add_error(report, -1, f"missing datasets: {missing}")
            return report
        n_records = len(h5["cif_tokenized"])
        report["n_records"] = n_records
        if n_records == 0:
            add_error(report, -1, "split contains no records")
            return report

        report["xrd_backend"] = (
            decode_string(h5["xrd_backend"][0])
            if "xrd_backend" in h5
            else "pymatgen"
        )
        if report["xrd_backend"] not in {"braggcalculator", "pymatgen"}:
            add_error(report, -1, f"unknown XRD backend {report['xrd_backend']!r}")

        lengths = np.asarray(h5["cif_token_length"], dtype=np.int64)
        report["token_lengths"] = {
            "min": int(lengths.min()),
            "p50": float(np.percentile(lengths, 50)),
            "p95": float(np.percentile(lengths, 95)),
            "p99": float(np.percentile(lengths, 99)),
            "max": int(lengths.max()),
        }
        block_size = int(config["block_size"])
        condition_tokens = (
            int(config.get("condition_n_tokens", 1))
            if config.get("condition") and not config.get("condition_cross_attention")
            else 0
        )
        max_model_tokens = int(lengths.max()) - 1 + condition_tokens
        report["max_model_tokens"] = max_model_tokens
        report["block_size"] = block_size
        report["batch_token_budget"] = int(config["batch_token_budget"])
        if int(lengths.min()) < 2:
            add_error(report, -1, "a record is shorter than two tokens")
        if int(lengths.max()) - 1 > block_size:
            add_error(report, -1, "maximum record length exceeds block_size + 1")
        if max_model_tokens > int(config["batch_token_budget"]):
            add_error(report, -1, "one record exceeds batch_token_budget")

        indices = sample_indices(n_records, max_items, seed)
        report["n_sampled"] = len(indices)
        for index in indices:
            validate_sample(h5, index, tokenizer, config, report)
    return report


def audit_dataset(config_path, dataset_dir=None, splits=("train", "val", "test"), max_items=100, seed=1337):
    config = load_config(config_path)
    dataset_dir = dataset_dir or config.get("dataset")
    if not dataset_dir:
        raise ValueError("dataset directory is missing from both arguments and config")
    reports = {
        split: audit_split(split_path(dataset_dir, split), config, max_items, seed + offset)
        for offset, split in enumerate(splits)
    }
    return {
        "status": "ok" if all(report["n_errors"] == 0 for report in reports.values()) else "failed",
        "config": os.path.abspath(config_path),
        "dataset": os.path.abspath(dataset_dir),
        "splits": reports,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit minicif_v2 data before training.")
    parser.add_argument("--config", required=True, help="minicif_v2 training YAML")
    parser.add_argument("--dataset-dir", default="", help="Override the dataset path in the config")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--max-items", type=int, default=100, help="Deep samples per split; 0 checks all")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", default="", help="Optional JSON output path")
    args = parser.parse_args()

    report = audit_dataset(
        args.config,
        dataset_dir=args.dataset_dir or None,
        splits=args.splits,
        max_items=args.max_items,
        seed=args.seed,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as handle:
            handle.write(rendered + "\n")
    raise SystemExit(0 if report["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
