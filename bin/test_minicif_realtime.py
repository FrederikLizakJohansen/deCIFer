#!/usr/bin/env python3

import argparse
import os
import random
import sys
from typing import Dict, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element

from decifer.decifer_model import Decifer, DeciferConfig
from decifer.minicif import START_TOKEN, MinicifTokenizer, minicif_to_structure, parse_minicif
from decifer.pxrd import discrete_to_continuous_xrd, nyquist_qstep
from bin.train import TrainConfig

PROMPT_MODE_ALIASES = {
    "pxrd": "start",
    "pxrd-elements": "formula",
    "pxrd-elements-cs": "formula-cs",
    "pxrd-elements-cs-sg": "formula-cs-sg",
}


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
    config = _config_to_dict(checkpoint.get("run_metadata", {}).get("config"))
    config.update(_config_to_dict(checkpoint.get("config")))
    return config


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


def clean_xrd_kwargs(config: Dict, args):
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


def read_sample(h5_path, index: Optional[int], seed: int):
    with h5py.File(h5_path, "r") as h5:
        n_items = len(h5["cif_tokenized"])
        if n_items == 0:
            raise ValueError(f"{h5_path} contains no rows")
        if index is None:
            index = random.Random(seed).randrange(n_items)
        if not 0 <= index < n_items:
            raise IndexError(f"--index must satisfy 0 <= index < {n_items}; got {index}")
        if "minicif_string" not in h5:
            raise KeyError(f"{h5_path} is missing minicif_string; use a minicif test.h5 file")
        minicif_string = h5["minicif_string"][index]
        if isinstance(minicif_string, bytes):
            minicif_string = minicif_string.decode("utf-8")
        q = np.asarray(h5["xrd_disc.q"][index], dtype=np.float32)
        iq = np.asarray(h5["xrd_disc.iq"][index], dtype=np.float32)
        name = h5["cif_name"][index] if "cif_name" in h5 else str(index)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
    return index, name, minicif_string, q, iq


def prompt_from_minicif(minicif_string, mode, tokenizer):
    mode = PROMPT_MODE_ALIASES.get(mode, mode)
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
    q_tensor = torch.tensor(q, dtype=torch.float32)
    iq_tensor = torch.tensor(iq, dtype=torch.float32)
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
    _, iq_cont, _ = continuous_from_sparse(q_disc.cpu().numpy(), iq_disc.cpu().numpy(), xrd_kwargs)
    return iq_cont


def generate_candidates(model, prompt, cond_vec, args, tokenizer):
    generated = []
    remaining = args.num_reps
    while remaining > 0:
        batch_size = min(args.generation_batch_size, remaining)
        batch_prompt = prompt.to(model.device).unsqueeze(0).repeat(batch_size, 1)
        batch_cond = None
        if model.config.condition:
            batch_cond = cond_vec.to(model.device).repeat(batch_size, 1)
        batch = model.generate_batched_reps(
            batch_prompt,
            args.max_new_tokens,
            cond_vec=batch_cond,
            start_indices_batch=[[0]] * batch_size,
            temperature=args.temperature,
            top_k=args.top_k,
            disable_pbar=not args.show_pbar,
            constrain_minicif=True,
        ).cpu().numpy()
        for ids in batch:
            ids = ids[ids != tokenizer.padding_id]
            generated.append(tokenizer.decode([int(token_id) for token_id in ids]))
        remaining -= batch_size
    return generated


def evaluate_candidates(candidates, reference_parsed, reference_structure, reference_iq, matcher, xrd_kwargs, wavelength):
    rows = []
    for rep, generated_minicif in enumerate(candidates):
        row = {
            "rep": rep,
            "parse_ok": False,
            "structure_ok": False,
            "structure_match": False,
            "composition_match": False,
            "space_group_match": False,
            "crystal_system_match": False,
            "rwp": None,
            "rmsd": None,
            "error": "",
            "generated_minicif": generated_minicif,
        }
        try:
            generated_parsed = parse_minicif(generated_minicif)
            row.update({
                "parse_ok": True,
                "space_group_match": generated_parsed.space_group == reference_parsed.space_group,
                "crystal_system_match": generated_parsed.crystal_system == reference_parsed.crystal_system,
            })
            generated_structure = minicif_to_structure(generated_minicif)
            generated_iq = structure_to_continuous_xrd(generated_structure, xrd_kwargs, wavelength)
            rmsd = matcher.get_rms_dist(reference_structure, generated_structure)
            rmsd_value = None if rmsd is None else float(rmsd[0])
            row.update({
                "structure_ok": True,
                "structure_match": rmsd_value is not None,
                "composition_match": generated_structure.composition.reduced_formula == reference_structure.composition.reduced_formula,
                "rwp": rwp(reference_iq, generated_iq),
                "rmsd": rmsd_value,
                "generated_iq": generated_iq,
                "generated_structure": generated_structure,
            })
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)
    return rows


def print_results(rows, print_minicifs):
    valid_rwps = [row["rwp"] for row in rows if row["rwp"] is not None]
    best = min((row for row in rows if row["rwp"] is not None), key=lambda row: row["rwp"], default=None)
    print("\nGenerated candidates")
    for row in rows:
        rwp_text = "nan" if row["rwp"] is None else f"{row['rwp']:.6f}"
        rmsd_text = "nan" if row["rmsd"] is None else f"{row['rmsd']:.6f}"
        print(
            f"[{row['rep']:03d}] parse={row['parse_ok']} structure={row['structure_ok']} "
            f"match={row['structure_match']} composition={row['composition_match']} "
            f"sg={row['space_group_match']} cs={row['crystal_system_match']} "
            f"Rwp={rwp_text} RMSD={rmsd_text}"
        )
        if row["error"]:
            print(f"      error: {row['error']}")
        if print_minicifs:
            print(f"      {row['generated_minicif']}")
    print("\nSummary")
    print(f"parse_rate={sum(row['parse_ok'] for row in rows) / max(len(rows), 1):.3f}")
    print(f"structure_rate={sum(row['structure_ok'] for row in rows) / max(len(rows), 1):.3f}")
    print(f"structure_match_rate={sum(row['structure_match'] for row in rows) / max(len(rows), 1):.3f}")
    print(f"valid_rwp_count={len(valid_rwps)}")
    if best is not None:
        print(f"best_rep={best['rep']} best_Rwp={best['rwp']:.6f}")


def best_row_by_rwp(rows):
    return min((row for row in rows if row["rwp"] is not None), key=lambda row: row["rwp"], default=None)


def save_fit_figure(path, q_grid, reference_iq, reference_structure, rows, sample_name, figure_supercell):
    best = best_row_by_rwp(rows)
    if best is None:
        raise ValueError("cannot create figure because no generated candidate produced a valid Rwp")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    generated_iq = best["generated_iq"]
    generated_structure = best["generated_structure"]
    residual = reference_iq - generated_iq

    fig = plt.figure(figsize=(14, 10), dpi=180, constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.35, 0.55, 2.0])
    ax_fit = fig.add_subplot(grid[0, :])
    ax_residual = fig.add_subplot(grid[1, :], sharex=ax_fit)
    ax_ref = fig.add_subplot(grid[2, 0], projection="3d")
    ax_gen = fig.add_subplot(grid[2, 1], projection="3d")

    ax_fit.plot(q_grid, reference_iq, color="#202124", linewidth=1.35, label="reference")
    ax_fit.plot(q_grid, generated_iq, color="#0072B2", linewidth=1.15, alpha=0.9, label=f"best generated, rep {best['rep']}")
    ax_fit.fill_between(q_grid, reference_iq, generated_iq, color="#0072B2", alpha=0.12, linewidth=0)
    ax_fit.set_ylabel("normalized intensity")
    ax_fit.set_title(f"{sample_name} PXRD fit, best Rwp={best['rwp']:.4f}")
    ax_fit.legend(frameon=False, loc="upper right")
    ax_fit.grid(alpha=0.18)

    ax_residual.axhline(0, color="#444444", linewidth=0.8)
    ax_residual.plot(q_grid, residual, color="#D55E00", linewidth=0.9)
    ax_residual.set_xlabel("q")
    ax_residual.set_ylabel("residual")
    ax_residual.grid(alpha=0.18)

    _plot_structure_panel(ax_ref, reference_structure, figure_supercell, "reference structure")
    _plot_structure_panel(ax_gen, generated_structure, figure_supercell, "best generated structure")

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_structure_panel(ax, structure, repeats, title):
    repeats = max(1, int(repeats))
    repeated = structure.copy()
    repeated.make_supercell([repeats, repeats, repeats])

    coords = np.asarray(repeated.cart_coords)
    species = [site.species_string for site in repeated]
    colors = {element: _element_color(element) for element in sorted(set(species))}
    for element in sorted(colors):
        element_coords = coords[[species_i == element for species_i in species]]
        ax.scatter(
            element_coords[:, 0],
            element_coords[:, 1],
            element_coords[:, 2],
            s=_element_marker_size(element),
            color=colors[element],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.92,
            label=element,
        )

    _draw_repeated_unit_cells(ax, structure.lattice.matrix, repeats)
    _set_axes_equal(ax, coords, structure.lattice.matrix, repeats)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=20, azim=35)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=8)
    ax.grid(False)


def _draw_repeated_unit_cells(ax, lattice_matrix, repeats):
    a_vec, b_vec, c_vec = np.asarray(lattice_matrix)
    edge_specs = [
        (np.zeros(3), a_vec),
        (np.zeros(3), b_vec),
        (np.zeros(3), c_vec),
        (a_vec, b_vec),
        (a_vec, c_vec),
        (b_vec, a_vec),
        (b_vec, c_vec),
        (c_vec, a_vec),
        (c_vec, b_vec),
        (a_vec + b_vec, c_vec),
        (a_vec + c_vec, b_vec),
        (b_vec + c_vec, a_vec),
    ]
    for i in range(repeats):
        for j in range(repeats):
            for k in range(repeats):
                origin = i * a_vec + j * b_vec + k * c_vec
                for start, delta in edge_specs:
                    points = np.vstack([origin + start, origin + start + delta])
                    ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#555555", linewidth=0.55, alpha=0.38)


def _set_axes_equal(ax, coords, lattice_matrix, repeats):
    a_vec, b_vec, c_vec = np.asarray(lattice_matrix)
    corners = []
    for i in [0, repeats]:
        for j in [0, repeats]:
            for k in [0, repeats]:
                corners.append(i * a_vec + j * b_vec + k * c_vec)
    all_points = np.vstack([coords, np.asarray(corners)])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    if radius == 0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))


def _element_color(element):
    palette = plt.get_cmap("tab20").colors
    try:
        index = Element(element).Z - 1
    except Exception:
        index = abs(hash(element))
    return palette[index % len(palette)]


def _element_marker_size(element):
    try:
        radius = Element(element).atomic_radius or 1.0
        return float(85 * max(0.6, min(radius, 1.8)))
    except Exception:
        return 85.0


def main():
    parser = argparse.ArgumentParser(description="Generate minicifs for one PXRD sample from a minicif HDF5 split.")
    parser.add_argument("--checkpoint", required=True, help="Path to ckpt.pt")
    parser.add_argument("--h5", required=True, help="Path to a minicif split HDF5 file, e.g. data/noma/serialized/test.h5")
    parser.add_argument("--index", type=int, default=None, help="Sample index in --h5. Defaults to a random row.")
    parser.add_argument("--num-reps", type=int, default=8, help="Number of minicifs to generate")
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--prompt-mode",
        choices=["pxrd", "pxrd-elements", "pxrd-elements-cs", "pxrd-elements-cs-sg", "start", "formula", "formula-cs", "formula-cs-sg"],
        default="pxrd",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-current", action="store_true", help="Use current_model instead of best_model_state")
    parser.add_argument("--qmin", type=float, default=None)
    parser.add_argument("--qmax", type=float, default=None)
    parser.add_argument("--qstep", type=float, default=None)
    parser.add_argument("--clean-fwhm", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--wavelength", default="CuKa")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-print-minicifs", action="store_true", help="Do not print generated minicif strings")
    parser.add_argument("--figure-out", default="", help="Optional path to save a PXRD/structure comparison figure")
    parser.add_argument("--figure-supercell", type=int, default=2, help="Supercell repeat count for structure panels")
    parser.add_argument("--show-pbar", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    checkpoint, model = load_checkpoint(args.checkpoint, device, use_best=not args.use_current)
    config = checkpoint_config(checkpoint)
    xrd_kwargs = clean_xrd_kwargs(config, args)
    tokenizer = MinicifTokenizer()

    index, name, reference_minicif, q_disc, iq_disc = read_sample(args.h5, args.index, args.seed)
    reference_parsed = parse_minicif(reference_minicif)
    reference_structure = minicif_to_structure(reference_minicif)
    q_grid, reference_iq, cond_iq = continuous_from_sparse(q_disc, iq_disc, xrd_kwargs)
    prompt = prompt_from_minicif(reference_minicif, args.prompt_mode, tokenizer)

    print(f"checkpoint: {os.path.abspath(args.checkpoint)}")
    print(f"h5: {os.path.abspath(args.h5)}")
    print(f"index: {index}")
    print(f"cif_name: {name}")
    print(f"device: {device}")
    print(f"prompt_mode: {args.prompt_mode}")
    print(f"xrd_kwargs: {xrd_kwargs}")
    print("\nReference canonical minicif")
    print(reference_minicif)

    candidates = generate_candidates(model, prompt, cond_iq, args, tokenizer)
    rows = evaluate_candidates(
        candidates,
        reference_parsed,
        reference_structure,
        reference_iq,
        StructureMatcher(),
        xrd_kwargs,
        args.wavelength,
    )
    print_results(rows, not args.no_print_minicifs)
    if args.figure_out:
        save_fit_figure(args.figure_out, q_grid, reference_iq, reference_structure, rows, name, args.figure_supercell)
        print(f"\nSaved figure: {os.path.abspath(args.figure_out)}")


if __name__ == "__main__":
    main()
