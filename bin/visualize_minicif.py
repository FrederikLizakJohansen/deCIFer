#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.structure_matcher import StructureMatcher
from tqdm.auto import tqdm

from decifer.decifer_dataset import DeciferDataset
from decifer.decifer_model import Decifer, DeciferConfig
from decifer.minicif import END_TOKEN, START_TOKEN, MinicifTokenizer, minicif_to_structure, parse_minicif
from decifer.pxrd import discrete_to_continuous_xrd, nyquist_qstep
from bin.train import TrainConfig


def rwp(reference, generated):
    reference = np.asarray(reference, dtype=float)
    generated = np.asarray(generated, dtype=float)
    return float(np.sqrt(np.sum((reference - generated) ** 2) / (np.sum(reference ** 2) + 1e-16)))


def load_checkpoint(path, device, use_best=True):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model_args = dict(checkpoint["model_args"])
    model = Decifer(DeciferConfig(**model_args)).to(device)
    state_key = "best_model_state" if use_best and checkpoint.get("best_model_state") is not None else "current_model"
    state_dict = checkpoint[state_key]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.device = device
    return checkpoint, model


def checkpoint_config(checkpoint):
    config = _config_to_dict(checkpoint.get("config"))
    metadata_config = _config_to_dict(checkpoint.get("run_metadata", {}).get("config"))
    merged = dict(metadata_config)
    merged.update(config)
    return merged


def _config_to_dict(config):
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "items"):
        return dict(config.items())
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return {}


def clean_xrd_kwargs(config, args):
    qmin = args.qmin if args.qmin is not None else float(config.get("qmin", 0.0))
    qmax = args.qmax if args.qmax is not None else float(config.get("qmax", 10.0))
    if args.qstep is not None:
        qstep = args.qstep
    elif float(config.get("nyquist_points_per_fwhm", 0.0)) > 0:
        qstep = nyquist_qstep(float(config.get("fwhm_range_min", 0.05)), float(config["nyquist_points_per_fwhm"]))
    else:
        qstep = float(config.get("qstep", 0.01))
    fwhm = args.clean_fwhm
    if fwhm is None:
        fwhm = 0.5 * (float(config.get("fwhm_range_min", 0.05)) + float(config.get("fwhm_range_max", 0.05)))
    eta = args.eta
    if eta is None:
        eta = 0.5 * (float(config.get("eta_range_min", 0.5)) + float(config.get("eta_range_max", 0.5)))
    return {
        "qmin": qmin,
        "qmax": qmax,
        "qstep": qstep,
        "fwhm_range": (fwhm, fwhm),
        "eta_range": (eta, eta),
        "noise_range": None,
        "intensity_scale_range": None,
        "mask_prob": None,
        "final_normalize": bool(config.get("final_normalize_xrd", True)),
    }


def dataset_path(dataset_dir, split):
    candidates = [
        os.path.join(dataset_dir, "serialized", f"{split}.h5"),
        os.path.join(dataset_dir, f"{split}.h5"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"could not find {split}.h5 under {dataset_dir}")


def prompt_from_minicif(minicif_string, mode, tokenizer):
    fields = minicif_string.strip().split()
    if mode == "start":
        prompt = START_TOKEN
    elif mode == "formula":
        stop = next(i for i, field in enumerate(fields) if field.startswith("cs_"))
        prompt = " ".join(fields[:stop])
    elif mode == "formula-cs":
        stop = next(i for i, field in enumerate(fields) if field.startswith("sg_"))
        prompt = " ".join(fields[:stop])
    elif mode == "formula-cs-sg":
        stop = fields.index("cell")
        prompt = " ".join(fields[:stop])
    else:
        raise ValueError(f"unknown prompt mode: {mode}")
    return torch.tensor(tokenizer.encode(tokenizer.tokenize_minicif(prompt)), dtype=torch.long)


def continuous_from_sparse(q, iq, xrd_kwargs):
    q_tensor = q if torch.is_tensor(q) else torch.tensor(q, dtype=torch.float32)
    iq_tensor = iq if torch.is_tensor(iq) else torch.tensor(iq, dtype=torch.float32)
    xrd = discrete_to_continuous_xrd(q_tensor.unsqueeze(0), iq_tensor.unsqueeze(0), **xrd_kwargs)
    return xrd["q"].cpu().numpy(), xrd["iq"][0].cpu().numpy(), xrd["iq"]


def structure_to_continuous_xrd(structure, xrd_kwargs, wavelength):
    calculator = XRDCalculator(wavelength=wavelength)
    qmax = xrd_kwargs["qmax"]
    max_q = ((4 * np.pi) / calculator.wavelength) * np.sin(np.radians(90))
    if qmax >= max_q:
        two_theta_range = None
    else:
        qmin = xrd_kwargs["qmin"]
        tth_min = np.degrees(2 * np.arcsin((qmin * calculator.wavelength) / (4 * np.pi)))
        tth_max = np.degrees(2 * np.arcsin((qmax * calculator.wavelength) / (4 * np.pi)))
        two_theta_range = (tth_min, tth_max)
    pattern = calculator.get_pattern(structure, two_theta_range=two_theta_range)
    theta = np.radians(pattern.x / 2)
    q_disc = torch.tensor(4 * np.pi * np.sin(theta) / calculator.wavelength, dtype=torch.float32)
    iq_disc = torch.tensor(pattern.y, dtype=torch.float32)
    iq_disc = iq_disc / (torch.max(iq_disc) + 1e-16)
    _, iq_cont, _ = continuous_from_sparse(q_disc, iq_disc, xrd_kwargs)
    return iq_cont


def generate_candidates(model, prompt, cond_vec, args, tokenizer):
    generated = []
    remaining = args.num_reps
    while remaining > 0:
        batch_size = min(args.generation_batch_size, remaining)
        batch_prompt = prompt.to(model.device).unsqueeze(0).repeat(batch_size, 1)
        batch_cond = cond_vec.to(model.device).repeat(batch_size, 1)
        batch = model.generate_batched_reps(
            batch_prompt,
            args.max_new_tokens,
            cond_vec=batch_cond,
            start_indices_batch=[[0]] * batch_size,
            temperature=args.temperature,
            top_k=args.top_k,
            disable_pbar=True,
            constrain_minicif=True,
        ).cpu().numpy()
        for ids in batch:
            ids = ids[ids != tokenizer.padding_id]
            generated.append(tokenizer.decode([int(token_id) for token_id in ids]))
        remaining -= batch_size
    return generated


def evaluate_split(split, h5_path, model, tokenizer, matcher, xrd_kwargs, args):
    dataset = DeciferDataset(h5_path, ["cif_name", "minicif_string", "cif_tokens", "xrd.q", "xrd.iq", "spacegroup", "crystal_system"])
    n_items = len(dataset) if args.max_items <= 0 else min(args.max_items, len(dataset))
    rows = []
    for sample_index in tqdm(range(n_items), desc=f"Evaluating {split}"):
        item = dataset[sample_index]
        reference_minicif = item["minicif_string"]
        try:
            reference_parsed = parse_minicif(reference_minicif)
            reference_structure = minicif_to_structure(reference_minicif)
            _, reference_iq, cond_iq = continuous_from_sparse(item["xrd.q"], item["xrd.iq"], xrd_kwargs)
            prompt = prompt_from_minicif(reference_minicif, args.prompt_mode, tokenizer)
            candidates = generate_candidates(model, prompt, cond_iq, args, tokenizer)
        except Exception as exc:
            rows.append({
                "split": split,
                "sample_index": sample_index,
                "cif_name": item["cif_name"],
                "rep": -1,
                "reference_error": str(exc),
                "parse_ok": False,
                "match": False,
            })
            continue

        for rep, generated_minicif in enumerate(candidates):
            row = {
                "split": split,
                "sample_index": sample_index,
                "cif_name": item["cif_name"],
                "rep": rep,
                "reference_minicif": reference_minicif,
                "generated_minicif": generated_minicif,
                "reference_space_group": reference_parsed.space_group,
                "reference_crystal_system": reference_parsed.crystal_system,
                "parse_ok": False,
                "match": False,
            }
            try:
                generated_parsed = parse_minicif(generated_minicif)
                generated_structure = minicif_to_structure(generated_minicif)
                generated_iq = structure_to_continuous_xrd(generated_structure, xrd_kwargs, args.wavelength)
                rmsd = matcher.get_rms_dist(reference_structure, generated_structure)
                rmsd_value = None if rmsd is None else float(rmsd[0])
                match = rmsd_value is not None
                if args.rmsd_threshold > 0 and rmsd_value is not None:
                    match = rmsd_value <= args.rmsd_threshold
                row.update({
                    "parse_ok": True,
                    "rwp": rwp(reference_iq, generated_iq),
                    "rmsd": rmsd_value,
                    "match": match,
                    "generated_space_group": generated_parsed.space_group,
                    "generated_crystal_system": generated_parsed.crystal_system,
                    "space_group_match": generated_parsed.space_group == reference_parsed.space_group,
                    "crystal_system_match": generated_parsed.crystal_system == reference_parsed.crystal_system,
                    "composition_match": generated_structure.composition.reduced_formula == reference_structure.composition.reduced_formula,
                })
            except Exception as exc:
                row["error"] = str(exc)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df):
    summaries = []
    for split, split_df in df.groupby("split", dropna=False):
        valid_rwp = split_df.dropna(subset=["rwp"]) if "rwp" in split_df else split_df.iloc[0:0]
        by_sample = split_df.groupby("sample_index")
        best_rwp = by_sample["rwp"].min() if "rwp" in split_df else pd.Series(dtype=float)
        summaries.append({
            "split": split,
            "n_samples": int(split_df["sample_index"].nunique()),
            "n_candidates": int(len(split_df[split_df["rep"] >= 0])),
            "parse_rate": float(split_df["parse_ok"].fillna(False).mean()),
            "candidate_match_rate": float(split_df["match"].fillna(False).mean()),
            "best_of_k_match_rate": float(by_sample["match"].max().fillna(False).mean()),
            "median_rwp": float(valid_rwp["rwp"].median()) if not valid_rwp.empty else np.nan,
            "median_best_rwp": float(best_rwp.median()) if not best_rwp.empty else np.nan,
            "mean_best_rwp": float(best_rwp.mean()) if not best_rwp.empty else np.nan,
            "median_matched_rmsd": float(split_df.loc[split_df["match"] == True, "rmsd"].median()) if "rmsd" in split_df else np.nan,
            "space_group_accuracy": float(split_df["space_group_match"].fillna(False).mean()) if "space_group_match" in split_df else np.nan,
            "crystal_system_accuracy": float(split_df["crystal_system_match"].fillna(False).mean()) if "crystal_system_match" in split_df else np.nan,
            "composition_match_rate": float(split_df["composition_match"].fillna(False).mean()) if "composition_match" in split_df else np.nan,
        })
    return pd.DataFrame(summaries)


def plot_learning_curves(checkpoint, out_dir):
    metrics = checkpoint.get("training_metrics") or {}
    epochs = metrics.get("epochs") or list(range(len(metrics.get("train_losses", []))))
    train_losses = metrics.get("train_losses", [])
    val_losses = metrics.get("val_losses", [])
    if not train_losses and not val_losses:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    if train_losses:
        ax.plot(epochs[:len(train_losses)], train_losses, label="train")
    if val_losses:
        ax.plot(epochs[:len(val_losses)], val_losses, label="validation")
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "learning_curves.png"))
    plt.close(fig)


def plot_metric_summary(summary, out_dir):
    if summary.empty:
        return
    metrics = ["parse_rate", "best_of_k_match_rate", "candidate_match_rate", "composition_match_rate", "space_group_accuracy", "crystal_system_accuracy"]
    available = [metric for metric in metrics if metric in summary.columns]
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
    x = np.arange(len(summary))
    width = 0.8 / max(1, len(available))
    for i, metric in enumerate(available):
        ax.bar(x + i * width, summary[metric], width=width, label=metric)
    ax.set_xticks(x + width * (len(available) - 1) / 2)
    ax.set_xticklabels(summary["split"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "metric_summary.png"))
    plt.close(fig)


def plot_rwp_distribution(df, out_dir):
    if "rwp" not in df or df["rwp"].dropna().empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    splits = list(df["split"].dropna().unique())
    values = [df.loc[df["split"] == split, "rwp"].dropna().to_numpy() for split in splits]
    ax.boxplot(values, labels=splits, showfliers=False)
    ax.set_ylabel("Rwp")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rwp_distribution.png"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate minicif learning-curve and validation/test evaluation reports.")
    parser.add_argument("--checkpoint", required=True, help="Path to ckpt.pt")
    parser.add_argument("--dataset-dir", default="", help="Dataset root containing serialized/{val,test}.h5; defaults to checkpoint config")
    parser.add_argument("--out-dir", default="", help="Output report directory; defaults to CHECKPOINT_DIR/minicif_report")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--max-items", type=int, default=0, help="Limit items per split; 0 means all")
    parser.add_argument("--num-reps", type=int, default=4, help="Generated candidates per dataset item")
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--prompt-mode", choices=["start", "formula", "formula-cs", "formula-cs-sg"], default="start")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-current", action="store_true", help="Use current_model instead of best_model_state")
    parser.add_argument("--rmsd-threshold", type=float, default=0.0, help="Optional positive RMSD threshold for match rate")
    parser.add_argument("--qmin", type=float, default=None)
    parser.add_argument("--qmax", type=float, default=None)
    parser.add_argument("--qstep", type=float, default=None)
    parser.add_argument("--clean-fwhm", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--wavelength", default="CuKa")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    checkpoint, model = load_checkpoint(args.checkpoint, device, use_best=not args.use_current)
    config = checkpoint_config(checkpoint)
    dataset_dir = args.dataset_dir or config.get("dataset")
    if not dataset_dir:
        raise ValueError("--dataset-dir is required when checkpoint config does not contain dataset")
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.checkpoint), "minicif_report")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = MinicifTokenizer()
    matcher = StructureMatcher()
    xrd_kwargs = clean_xrd_kwargs(config, args)
    plot_learning_curves(checkpoint, out_dir)

    frames = []
    for split in args.splits:
        path = dataset_path(dataset_dir, split)
        frames.append(evaluate_split(split, path, model, tokenizer, matcher, xrd_kwargs, args))
    results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = summarize(results)

    results.to_csv(os.path.join(out_dir, "minicif_generation_metrics.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "minicif_summary.csv"), index=False)
    with open(os.path.join(out_dir, "minicif_summary.json"), "w") as f:
        json.dump({
            "checkpoint": os.path.abspath(args.checkpoint),
            "dataset_dir": os.path.abspath(dataset_dir),
            "xrd_kwargs": xrd_kwargs,
            "summary": summary.to_dict(orient="records"),
        }, f, indent=2)
    plot_metric_summary(summary, out_dir)
    plot_rwp_distribution(results, out_dir)
    print(summary.to_string(index=False))
    print(f"Wrote minicif report to {out_dir}")


if __name__ == "__main__":
    main()
