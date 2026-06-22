#!/usr/bin/env python3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- standard library ---
import os
import re
import io
import pickle
import importlib.util
from glob import glob
from collections import Counter
from typing import Optional, Union, Tuple

# --- third-party: numerical / ML ---
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

# --- plotting ---
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.patches import Patch, Circle, FancyArrowPatch, PathPatch
import matplotlib.patheffects as path_effects

# --- crystallography / materials ---
from pymatgen.core import Structure
from pymatgen.core import Structure as PMGStructure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ase.visualize.plot import plot_atoms
from ase.data import colors as asecolors, atomic_numbers

# --- project-specific ---

from decifer.utility import (
    pxrd_from_cif,
    extract_formula_nonreduced,
    extract_space_group_symbol,
    extract_numeric_property,
)
import argparse
import gc

plt.show = lambda *args, **kwargs: None

_ABLATION_STATS_MODULE = None


def save_pdf_and_png(fig, savepath, *, dpi=150, facecolor="white", **kwargs):
    path = Path(savepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi, facecolor=facecolor, **kwargs)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=dpi, facecolor=facecolor, **kwargs)


def load_ablation_statistics_module():
    global _ABLATION_STATS_MODULE
    if _ABLATION_STATS_MODULE is None:
        module_path = REPO_ROOT / "bin" / "ablation_statistics.py"
        spec = importlib.util.spec_from_file_location("ablation_statistics_runtime", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        _ABLATION_STATS_MODULE = module
    return _ABLATION_STATS_MODULE


def build_ci_summary_df_from_runs(
    runs,
    parameter_keys,
    *,
    tag_set=("a", "b", "c"),
    n_boot=10000,
    confidence=0.95,
    seed=0,
):
    abstat = load_ablation_statistics_module()
    rng = np.random.default_rng(seed)
    rows = []
    for label, conditioned, unconditioned in runs:
        for param_key in parameter_keys:
            cond_conditions = conditioned.get(param_key, {})
            nocond_conditions = unconditioned.get(param_key, {})
            shared_keys = sorted(set(cond_conditions).intersection(nocond_conditions), key=abstat.sort_key)
            for condition_key in shared_keys:
                metrics = abstat.compute_condition_metrics(
                    cond_conditions[condition_key],
                    nocond_conditions[condition_key],
                    tag_set=tag_set,
                    n_boot=n_boot,
                    confidence=confidence,
                    rng=rng,
                )
                row = {
                    "run_label": label,
                    "parameter_key": param_key,
                    "condition_key": str(condition_key),
                    "condition_value_numeric": abstat._maybe_float(condition_key),
                }
                row.update(metrics)
                rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values(
        ["parameter_key", "run_label", "condition_value_numeric", "condition_key"]
    )

def plot_unit_cell_with_boundaries(
    structure,
    ax=None,
    tol=1e-5,
    radii=0.8,
    rotation=("45x, -15y, 90z"),
    offset=(0, 0, 0),
):
    """
    Plot the unit cell plus image atoms lying exactly on the +x/+y/+z boundaries.
    """
    if ax is None:
        fig, ax = plt.subplots()

    translation_vectors = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]

    all_species, all_coords = [], []

    for tv in translation_vectors:
        tv_cart = structure.lattice.get_cartesian_coords(tv)
        for site in structure:
            if tv == [0, 0, 0]:
                all_species.append(site.species_string)
                all_coords.append(site.coords)
            else:
                if all(
                    site.frac_coords[i] < tol
                    for i, shift in enumerate(tv)
                    if shift == 1
                ):
                    all_species.append(site.species_string)
                    all_coords.append(site.coords + tv_cart)

    all_coords = np.array(all_coords)
    min_coords, max_coords = all_coords.min(0), all_coords.max(0)
    lattice_vectors = np.diag(max_coords - min_coords)

    discrete_structure = PMGStructure(
        lattice=lattice_vectors,
        species=all_species,
        coords=all_coords,
        coords_are_cartesian=True,
    )

    ase_atoms = AseAtomsAdaptor.get_atoms(discrete_structure)
    ase_atoms.set_pbc([False, False, False])
    plot_atoms(
        ase_atoms,
        ax,
        radii=radii,
        show_unit_cell=True,
        rotation=rotation,
        offset=offset,
    )

    return ax, discrete_structure


def ablation_fig(
    results,
    uresults,
    param_keys,
    savepath=None,
    dpi=100,
    figscale=1,
    figlen=10,
    wspace_outer=0.2,
    hspace_outer=0.1,
    wspace_left=0.1,
    show_AB=True,
    show_u=False,
    fig_height_factor=2.25,
    cell_mins=None,
    cell_maxs=None,
    bbox_left=(0.75, 1.2),
    bbox_right=(0.75, 1.2),
    ncol_left=1,
    ncol_right=1,
    AB_x=[6.0, 6.0, 6.0],
    cell_tag_sets=[["a", "b", "c"]],
    title_left=None,
    title_left_pad=0.0,
    show_mid=True,
    plot_ubest=False,
    show_mean=True,
    show_best=True,
    show_boxplots=False,
    ytick_size=None,
    AB_symbols=["A", "B"],
    left_xlabels=None,
    mid_size=0.6,
    show_best_stem=True,
    swap_left_xy=True,
    width_ratios=[1.0, 1.0],
    x_txt=0.75,
    pxrd_xlim=(0, 8),
    best_markersize=50,
    pred_ms=2,
    example_idxs_results=None,
    example_idxs_uresults=None,
    # NEW: Parameters for discrete/categorical handling
    xlabel_rotation=0,           # Rotation angle for x-tick labels
    pxrd_annotation_key=None,    # Key name to use in PXRD annotations (e.g., "noise", "shift")
                                 # If None, will use "η" for numeric, key name for discrete
    panel_labels=None,           # e.g. ["(a)", "(b)", "(c)", "(d)"]: annotate the four
                                 # quadrants (top-left, top-right, bottom-left, bottom-right)
    panel_label_fontsize=10,
):
    """
    Make the "ablation" summary plot with split subplots for results and uresults.
    
    Parameters:
    -----------
    example_idxs_results : list, optional
        Example indices to use for results plots. If None, uses param["example_idxs"]
    example_idxs_uresults : list, optional
        Example indices to use for uresults plots. If None, uses param["example_idxs"]
    xlabel_rotation : float, optional
        Rotation angle for x-axis tick labels (useful for discrete parameters)
    pxrd_annotation_key : str, optional
        Key name to display in PXRD annotations. If None, uses "η" for numeric params
        or the parameter key name for discrete/non-numeric params.
    """

    def extract_pxrd(data, p_key, ex_key):
        exp = data[p_key][ex_key]["best_experiment"]
        ref = exp["pxrd_ref"]
        gen_clean = exp["pxrd_gen_clean"]
        return ref["q"], ref["iq"], gen_clean["q_disc"][0], gen_clean["iq_disc"][0]

    def compute_best_lengths(data, p_key, p_val_keys, tag_set):
        return [
            np.mean(
                [
                    extract_numeric_property(
                        cif,
                        (
                            f"_cell_length_{dim}"
                            if dim in ["a", "b", "c"]
                            else f"_cell_angle_{dim}"
                        ),
                    )
                    for dim in tag_set
                ]
            )
            for cif in (
                data[p_key][k]["best_experiment"]["generated_cif"] for k in p_val_keys
            )
        ]

    def compute_stats(data_section, p_key, p_val_keys, tag_set, handle_empty=False):
        all_L, all_A = [], []
        for k in p_val_keys:
            exps = data_section[p_key][k]["experiments"]
            lengths = [
                extract_numeric_property(
                    e["generated_cif"],
                    f"_cell_length_{d}" if d in ["a", "b", "c"] else f"_cell_angle_{d}",
                )
                for e in exps
                for d in tag_set
            ]
            angles = [
                extract_numeric_property(e["generated_cif"], f"_cell_angle_{d}")
                for e in exps
                for d in ["alpha", "beta", "gamma"]
            ]
            all_L.append(lengths)
            all_A.append(angles)

        meanf = lambda x: np.mean(x) if x else np.mean(np.concatenate(all_L))
        stdf = lambda x: np.std(x) if x else 0.0

        return (
            np.array([meanf(L) for L in all_L]),
            np.array([stdf(L) for L in all_L]),
            np.array([meanf(A) for A in all_A]),
            np.array([stdf(A) for A in all_A]),
        )

    def compute_match_stats(data_section, p_key, p_val_keys, handle_empty=False):
        all_R = []
        for k in p_val_keys:
            exps = data_section[p_key][k]["experiments"]
            all_R.append([e["structure_match"] for e in exps])

        meanf = lambda r: np.mean(r) if r else 0.0
        stdf = lambda r: np.std(r) if r else 0.0
        maxf = lambda r: np.max(r) if r else 0.0

        return (
            np.array([meanf(r) for r in all_R]),
            np.array([stdf(r) for r in all_R]),
            np.array([maxf(r) for r in all_R]),
        )

    def format_annotation_value(val, key, is_numeric):
        """Format the value for PXRD annotation based on type."""
        if is_numeric:
            try:
                return f"{float(val):.2g}"
            except:
                return str(val)
        else:
            # For discrete/categorical, truncate if too long
            s = str(val)
            if len(s) > 12:
                return s[:10] + "..."
            return s

    # ------------------------------------------------------------------ #
    # Plot-style constants - UPDATED COLORS
    # ------------------------------------------------------------------ #
    colors_picks = {
        "exp": "#333333",           # Dark gray for experimental data
        "pred": "#DC143C",          # Blue for deCIFer predictions
        "pred_u": "#008080",        # Orange for U-deCIFer predictions
        "ref": "#333333",           # Dark gray for reference
    }
    peak_scale = 0.8
    n_params = len(param_keys)

    num_length_plots = len(cell_tag_sets)
    if left_xlabels is not None and len(left_xlabels) != num_length_plots:
        raise ValueError("left_xlabels must have the same length as cell_tag_sets")

    # --- Band styling (color + alpha, NO hatching)
    band_alpha = 0.18
    band_edgecolor = "none"
    band_lw = 0.0

    def band_fill(
        ax,
        *,
        x=None,
        y1=None,
        y2=None,
        y=None,
        x1=None,
        x2=None,
        facecolor="C0",
        alpha=0.2,
        edgecolor="k",
        lw=0.8,
        zorder=1,
    ):
        """
        Draw an uncertainty band with colored semi-transparent fill and outline.
        NO hatching.
        """
        if x is not None:
            ax.fill_between(
                x,
                y1,
                y2,
                facecolor=facecolor,
                alpha=alpha,
                edgecolor=edgecolor,
                linewidth=lw,
                zorder=zorder,
            )
        else:
            ax.fill_betweenx(
                y,
                x1,
                x2,
                facecolor=facecolor,
                alpha=alpha,
                edgecolor=edgecolor,
                linewidth=lw,
                zorder=zorder,
            )

    # ------------------------------------------------------------------ #
    # Figure & GridSpec layout - SPLIT INTO TWO ROWS
    # ------------------------------------------------------------------ #
    fig = plt.figure(
        figsize=(figlen * figscale, (fig_height_factor * n_params * 2) * figscale), dpi=dpi
    )

    # Main outer grid: 2 rows (results, uresults), 2 columns (left panels, right PXRD)
    outer = gridspec.GridSpec(2, 2, width_ratios=width_ratios, wspace=wspace_outer, hspace=hspace_outer)

    left_cols = num_length_plots + (1 if show_mid else 0)
    left_widths = [1] * num_length_plots + ([mid_size] if show_mid else [])
    
    # Top row (results)
    gs_left_top = gridspec.GridSpecFromSubplotSpec(
        n_params,
        left_cols,
        subplot_spec=outer[0, 0],
        width_ratios=left_widths,
        wspace=wspace_left,
    )
    gs_right_top = gridspec.GridSpecFromSubplotSpec(n_params, 1, subplot_spec=outer[0, 1])

    # Bottom row (uresults)
    gs_left_bottom = gridspec.GridSpecFromSubplotSpec(
        n_params,
        left_cols,
        subplot_spec=outer[1, 0],
        width_ratios=left_widths,
        wspace=wspace_left,
    )
    gs_right_bottom = gridspec.GridSpecFromSubplotSpec(n_params, 1, subplot_spec=outer[1, 1])

    total_cols = left_cols + 1  # +1 for PXRD column
    axes_results = np.empty((n_params, total_cols), dtype=object)
    axes_uresults = np.empty((n_params, total_cols), dtype=object)

    # Create axes for results (top)
    for i in range(n_params):
        for j in range(num_length_plots):
            axes_results[i, j] = fig.add_subplot(gs_left_top[i, j])
        if show_mid:
            axes_results[i, num_length_plots] = fig.add_subplot(gs_left_top[i, num_length_plots])
        axes_results[i, -1] = fig.add_subplot(gs_right_top[i, 0])

    # Create axes for uresults (bottom)
    for i in range(n_params):
        for j in range(num_length_plots):
            axes_uresults[i, j] = fig.add_subplot(gs_left_bottom[i, j])
        if show_mid:
            axes_uresults[i, num_length_plots] = fig.add_subplot(gs_left_bottom[i, num_length_plots])
        axes_uresults[i, -1] = fig.add_subplot(gs_right_bottom[i, 0])

    if panel_labels:
        quadrant_axes = [
            axes_results[0, 0],
            axes_results[0, -1],
            axes_uresults[0, 0],
            axes_uresults[0, -1],
        ]
        for ax_quad, label in zip(quadrant_axes, panel_labels):
            ax_quad.text(
                0.02,
                0.97,
                label,
                transform=ax_quad.transAxes,
                ha="left",
                va="top",
                fontsize=panel_label_fontsize,
                color="black",
                zorder=10,
            )

    # ------------------------------------------------------------------ #
    # MAIN LOOP over parameters - RESULTS (TOP)
    # ------------------------------------------------------------------ #
    for i, param in enumerate(param_keys):
        p_key = param["key"]
        p_xlabel = param["xlabel"]
        p_unit = param.get("unit", "")

        # parameter values (numeric or categorical)
        p_val_keys = list(results[p_key].keys())
        is_num, num_vals = True, []
        for k in p_val_keys:
            try:
                num_vals.append(float(k))
            except ValueError:
                is_num = False
                break
        param_vals = num_vals if is_num else list(range(len(p_val_keys)))

        # Determine annotation key for PXRD
        if pxrd_annotation_key is not None:
            annot_key = pxrd_annotation_key
        elif is_num:
            annot_key = "η"
        else:
            # Use a shortened version of the parameter key
            key_map = {
                "chebychev_norm_coeffs": "bg",
                "preferred_orientation_range": "orient",
                "mask_ranges": "mask",
                "noise": "noise",
                "q_shift": "shift",
                "base_fwhm": "FWHM",
            }
            annot_key = key_map.get(p_key, p_key[:6])

        # example indices - use independent parameter if provided
        if example_idxs_results is not None:
            ex_idxs = example_idxs_results
        else:
            ex_idxs = param.get("example_idxs", [0, 1])
        
        # Ensure indices are within bounds
        ex_idxs = [idx for idx in ex_idxs if idx < len(p_val_keys)]
        example_keys = [p_val_keys[idx] for idx in ex_idxs]
        example_param_vals = np.array(param_vals)[ex_idxs]

        # extend AB_symbols list if necessary
        if len(AB_symbols) < len(example_keys):
            import string
            extras = [s for s in string.ascii_uppercase if s not in AB_symbols]
            AB_symbols.extend(extras[: len(example_keys) - len(AB_symbols)])

        # ========== PXRD overlay (right column) - RESULTS ==========================
        ax_pxrd = axes_results[i, -1]
        N = len(example_keys)
        spacing = 1.0
        offsets = [(j - (N - 1)) * spacing for j in range(N)]
        annot_off = spacing * 0.5

        for j, ex_key in enumerate(example_keys):
            q, iq, qd, iqd = extract_pxrd(results, p_key, ex_key)
            off = offsets[j]

            ax_pxrd.plot(
                q,
                iq * peak_scale + off,
                lw=1,
                color=colors_picks["exp"],
                label=("PXRD" if j == 0 and i == 0 else None),
            )

            if show_best_stem:
                stem = ax_pxrd.stem(
                    qd,
                    iqd / 100 * peak_scale + off,
                    linefmt="-",
                    markerfmt="o",
                    basefmt=" ",
                    bottom=off,
                    label=("Best XRD pred. (deCIFer)" if j == 0 and i == 0 else None),
                )
                stem.markerline.set(
                    markersize=pred_ms,
                    alpha=0.8,
                    markerfacecolor="white",
                    markeredgecolor=colors_picks["pred"],
                    markeredgewidth=1,
                )
                stem.stemlines.set(color=colors_picks["pred"], linewidth=1, alpha=0.5)

            # Format annotation based on numeric or discrete
            annot_val = format_annotation_value(ex_key, p_key, is_num)
            y_txt = off + annot_off + 0.25
            ax_pxrd.text(
                x_txt,
                y_txt,
                f"{annot_key} = {annot_val}",
                ha="center",
                va="top",
                fontsize=8,
                path_effects=[path_effects.withStroke(linewidth=3, foreground="white")],
            )

        ax_pxrd.set(yticks=[], ylabel=r"$I(Q)$ [a.u.]", xlim=pxrd_xlim)
        
        # Remove x-tick labels for all top row plots
        ax_pxrd.set_xticklabels([])
        
        ax_pxrd.yaxis.tick_right()
        ax_pxrd.yaxis.set_label_position("right")
        
        # Clean up spines
        ax_pxrd.spines['top'].set_visible(False)
        ax_pxrd.spines['left'].set_visible(False)

        # ========== LENGTH PANELS (one per tag-set) - RESULTS ======================
        for ts_idx, tag_set in enumerate(cell_tag_sets):
            ax_len = axes_results[i, ts_idx]

            best_lens = compute_best_lengths(results, p_key, p_val_keys, tag_set)
            mL, sL, _, _ = compute_stats(results, p_key, p_val_keys, tag_set)

            # compute y-positions for each example
            if is_num:
                example_vals = [float(k) for k in example_keys]
            else:
                example_vals = ex_idxs

            # reference line
            ref_means = [
                np.mean(
                    [
                        getattr(
                            results[p_key][k]["best_experiment"][
                                "reference_structure"
                            ].lattice,
                            d,
                        )
                        for d in tag_set
                    ]
                )
                for k in p_val_keys
            ]
            if swap_left_xy:
                ax_len.plot(
                    param_vals,
                    ref_means,
                    linestyle=":",
                    color=colors_picks["ref"],
                    linewidth=1,
                    #label=("Reference" if i == 0 and ts_idx == 0 else None),
                )
            else:
                ax_len.plot(
                    ref_means,
                    param_vals,
                    linestyle=":",
                    color=colors_picks["ref"],
                    linewidth=1,
                    #label=("Reference" if i == 0 and ts_idx == 0 else None),
                )

            # --- mean ± std band (NO hatching)
            if swap_left_xy:
                if show_mean:
                    band_fill(
                        ax_len,
                        x=param_vals,
                        y1=mL - sL,
                        y2=mL + sL,
                        facecolor=colors_picks["pred"],
                        alpha=band_alpha,
                        edgecolor=band_edgecolor,
                        lw=band_lw,
                        zorder=1,
                    )
                if show_best:
                    ax_len.scatter(
                        param_vals,
                        best_lens,
                        color=colors_picks["pred"],
                        s=best_markersize,
                        edgecolors=colors_picks["pred"],
                        facecolors="white",
                        lw=1,
                        zorder=9,
                        alpha=0.75,
                        label=("Best XRD pred. (deCIFer)" if i == 0 and ts_idx == 0 else None),
                    )
            else:
                if show_mean:
                    band_fill(
                        ax_len,
                        y=param_vals,
                        x1=mL - sL,
                        x2=mL + sL,
                        facecolor=colors_picks["pred"],
                        alpha=band_alpha,
                        edgecolor=band_edgecolor,
                        lw=band_lw,
                        zorder=1,
                    )
                if show_best:
                    ax_len.scatter(
                        best_lens,
                        param_vals,
                        color=colors_picks["pred"],
                        s=best_markersize,
                        edgecolors=colors_picks["pred"],
                        facecolors="white",
                        lw=1,
                        zorder=9,
                        alpha=0.75,
                        label=("Best pred. (deCIFer)" if i == 0 and ts_idx == 0 else None),
                    )

            # Axis labels and ticks
            if swap_left_xy:
                # Parameter on x-axis, cell lengths on y-axis
                if ts_idx == 0:
                    ax_len.set_ylabel(f"{p_xlabel}{p_unit}")
                    if not is_num:
                        ax_len.set_xticks(param_vals)
                        ax_len.set_xticklabels(p_val_keys, rotation=xlabel_rotation,
                                               ha='right' if xlabel_rotation > 0 else 'center',
                                               fontsize=8)
                    if ytick_size:
                        ax_len.tick_params(axis="y", labelsize=ytick_size)
                else:
                    ax_len.set_ylabel("")
                    ax_len.tick_params(axis="y", which="both", labelleft=False)
            else:
                # Parameter on y-axis, cell lengths on x-axis
                if ts_idx == 0:
                    ax_len.set_ylabel(f"{p_xlabel}{p_unit}")
                    if not is_num:
                        ax_len.set_yticks(param_vals)
                        ax_len.set_yticklabels(p_val_keys, fontsize=8)
                    if ytick_size:
                        ax_len.tick_params(axis="y", labelsize=ytick_size)
                else:
                    ax_len.set_ylabel("")
                    ax_len.tick_params(axis="y", which="both", labelleft=False)

            # Remove x-tick labels for all top row plots
            ax_len.set_xticklabels([])

            # example A/B horizontal/vertical bars (& labels) once per row
            if ts_idx == 0:
                for j, ev in enumerate(example_vals):
                    if show_AB:
                        if swap_left_xy:
                            ax_len.axvline(ev, color="k", lw=0.5, ls="-", alpha=0.5)
                            ax_len.text(
                                ev,
                                ax_len.get_ylim()[1] * 0.95,
                                AB_symbols[j],
                                ha="center",
                                va="top",
                                fontsize=10,
                                path_effects=[
                                    path_effects.withStroke(linewidth=3, foreground="white")
                                ],
                            )
                        else:
                            ax_len.axhline(ev, color="k", lw=0.5, ls="-", alpha=0.5)
                            ax_len.text(
                                AB_x[i] if i < len(AB_x) else AB_x[-1],
                                ev,
                                AB_symbols[j],
                                ha="center",
                                va="center",
                                fontsize=12,
                                path_effects=[
                                    path_effects.withStroke(linewidth=3, foreground="white")
                                ],
                            )
            else:
                for ev in example_vals:
                    if swap_left_xy:
                        ax_len.axvline(ev, color="k", lw=0.5, ls="-", alpha=0.5)
                    else:
                        ax_len.axhline(ev, color="k", lw=0.5, ls="-", alpha=0.5)
            
            # Clean up spines
            ax_len.spines['top'].set_visible(False)
            ax_len.spines['right'].set_visible(False)

        # ========== MATCH-RATE column (optional) - RESULTS =========================
        if show_mid:
            ax_mid = axes_results[i, num_length_plots]
            mR, _, _ = compute_match_stats(results, p_key, p_val_keys)

            if swap_left_xy:
                ax_mid.scatter(
                    param_vals,
                    mR,
                    color=colors_picks["pred"],
                    s=20,
                    edgecolors=colors_picks["pred"],
                    facecolors="white",
                    lw=1,
                    zorder=9,
                    alpha=0.75,
                )
                ax_mid.set_ylim(-0.1, 1.1)
                ax_mid.set_xticklabels([])
            else:
                ax_mid.scatter(
                    mR,
                    param_vals,
                    color=colors_picks["pred"],
                    s=20,
                    edgecolors=colors_picks["pred"],
                    facecolors="white",
                    lw=1,
                    zorder=9,
                    alpha=0.75,
                )
                ax_mid.set_xlim(-0.1, 1.1)
                ax_mid.set_xticklabels([])
            
            ax_mid.set_yticklabels([])
            ax_mid.grid(axis="x" if not swap_left_xy else "y", alpha=0.5)
            ax_mid.spines['top'].set_visible(False)
            ax_mid.spines['right'].set_visible(False)

    # ------------------------------------------------------------------ #
    # MAIN LOOP over parameters - URESULTS (BOTTOM)
    # ------------------------------------------------------------------ #
    for i, param in enumerate(param_keys):
        p_key = param["key"]
        p_xlabel = param["xlabel"]
        p_unit = param.get("unit", "")

        p_val_keys = list(results[p_key].keys())
        is_num, num_vals = True, []
        for k in p_val_keys:
            try:
                num_vals.append(float(k))
            except ValueError:
                is_num = False
                break
        param_vals = num_vals if is_num else list(range(len(p_val_keys)))

        # Determine annotation key for PXRD
        if pxrd_annotation_key is not None:
            annot_key = pxrd_annotation_key
        elif is_num:
            annot_key = "η"
        else:
            key_map = {
                "chebychev_norm_coeffs": "bg",
                "preferred_orientation_range": "orient",
                "mask_ranges": "mask",
                "noise": "noise",
                "q_shift": "shift",
                "base_fwhm": "FWHM",
            }
            annot_key = key_map.get(p_key, p_key[:6])

        # example indices - use independent parameter if provided
        if example_idxs_uresults is not None:
            ex_idxs = example_idxs_uresults
        else:
            ex_idxs = param.get("example_idxs", [0, 1])
        
        # Ensure indices are within bounds
        ex_idxs = [idx for idx in ex_idxs if idx < len(p_val_keys)]
        example_keys = [p_val_keys[idx] for idx in ex_idxs]
        example_param_vals = np.array(param_vals)[ex_idxs]

        if len(AB_symbols) < len(example_keys):
            import string
            extras = [s for s in string.ascii_uppercase if s not in AB_symbols]
            AB_symbols.extend(extras[: len(example_keys) - len(AB_symbols)])

        # ========== PXRD overlay (right column) - URESULTS ==========================
        ax_pxrd = axes_uresults[i, -1]
        N = len(example_keys)
        spacing = 1.0
        offsets = [(j - (N - 1)) * spacing for j in range(N)]
        annot_off = spacing * 0.5

        for j, ex_key in enumerate(example_keys):
            q, iq, qd, iqd = extract_pxrd(uresults, p_key, ex_key)
            off = offsets[j]

            ax_pxrd.plot(
                q,
                iq * peak_scale + off,
                lw=1,
                color=colors_picks["exp"],
                label=None,
            )

            if show_best_stem:
                stem = ax_pxrd.stem(
                    qd,
                    iqd / 100 * peak_scale + off,
                    linefmt="-",
                    markerfmt="o",
                    basefmt=" ",
                    bottom=off,
                    label=("Best pred. (U-deCIFer)" if j == 0 and i == 0 else None),
                )
                stem.markerline.set(
                    markersize=pred_ms,
                    alpha=0.8,
                    markerfacecolor="white",
                    markeredgecolor=colors_picks["pred_u"],
                    markeredgewidth=1,
                )
                stem.stemlines.set(color=colors_picks["pred_u"], linewidth=1, alpha=0.5)

            # Format annotation based on numeric or discrete
            annot_val = format_annotation_value(ex_key, p_key, is_num)
            y_txt = off + annot_off + 0.25
            ax_pxrd.text(
                x_txt,
                y_txt,
                f"{annot_key} = {annot_val}",
                ha="center",
                va="top",
                fontsize=8,
                path_effects=[path_effects.withStroke(linewidth=3, foreground="white")],
            )

        ax_pxrd.set(yticks=[], ylabel=r"$I(Q)$ [a.u.]", xlim=pxrd_xlim)
        
        # Only add x-label at bottom
        if i == n_params - 1:
            ax_pxrd.set_xlabel(r"$Q$ [Å$^{-1}$]")
        else:
            ax_pxrd.set_xticklabels([])
            
        ax_pxrd.yaxis.tick_right()
        ax_pxrd.yaxis.set_label_position("right")
        
        # Clean up spines
        ax_pxrd.spines['top'].set_visible(False)
        ax_pxrd.spines['left'].set_visible(False)

        # ========== LENGTH PANELS (one per tag-set) - URESULTS ======================
        for ts_idx, tag_set in enumerate(cell_tag_sets):
            ax_len = axes_uresults[i, ts_idx]

            best_lens = compute_best_lengths(uresults, p_key, p_val_keys, tag_set)
            mL, sL, _, _ = compute_stats(uresults, p_key, p_val_keys, tag_set, handle_empty=True)

            if is_num:
                example_vals = [float(k) for k in example_keys]
            else:
                example_vals = ex_idxs

            # reference line
            ref_means = [
                np.mean(
                    [
                        getattr(
                            uresults[p_key][k]["best_experiment"][
                                "reference_structure"
                            ].lattice,
                            d,
                        )
                        for d in tag_set
                    ]
                )
                for k in p_val_keys
            ]
            if swap_left_xy:
                ax_len.plot(
                    param_vals,
                    ref_means,
                    linestyle=":",
                    color=colors_picks["ref"],
                    linewidth=1,
                )
            else:
                ax_len.plot(
                    ref_means,
                    param_vals,
                    linestyle=":",
                    color=colors_picks["ref"],
                    linewidth=1,
                )

            # --- mean ± std band (NO hatching)
            if swap_left_xy:
                if show_mean:
                    band_fill(
                        ax_len,
                        x=param_vals,
                        y1=mL - sL,
                        y2=mL + sL,
                        facecolor=colors_picks["pred_u"],
                        alpha=band_alpha,
                        edgecolor=band_edgecolor,
                        lw=band_lw,
                        zorder=1,
                    )
                if show_best:
                    ax_len.scatter(
                        param_vals,
                        best_lens,
                        color=colors_picks["pred_u"],
                        marker="^",
                        s=best_markersize,
                        edgecolors=colors_picks["pred_u"],
                        facecolors="white",
                        lw=1,
                        zorder=9,
                        alpha=0.75,
                        label=("Best pred. (U-deCIFer)" if i == 0 and ts_idx == 0 else None),
                    )
            else:
                if show_mean:
                    band_fill(
                        ax_len,
                        y=param_vals,
                        x1=mL - sL,
                        x2=mL + sL,
                        facecolor=colors_picks["pred_u"],
                        alpha=band_alpha,
                        edgecolor=band_edgecolor,
                        lw=band_lw,
                        zorder=1,
                    )
                if show_best:
                    ax_len.scatter(
                        best_lens,
                        param_vals,
                        color=colors_picks["pred_u"],
                        marker="^",
                        s=best_markersize,
                        edgecolors=colors_picks["pred_u"],
                        facecolors="white",
                        lw=1,
                        zorder=9,
                        alpha=0.75,
                        label=("Best pred. (U-deCIFer)" if i == 0 and ts_idx == 0 else None),
                    )

            # Axis labels and ticks
            if swap_left_xy:
                # Parameter on x-axis, cell lengths on y-axis
                if ts_idx == 0:
                    ax_len.set_ylabel(f"{p_xlabel}{p_unit}")
                    if not is_num:
                        ax_len.set_xticks(param_vals)
                        ax_len.set_xticklabels(p_val_keys, rotation=xlabel_rotation,
                                               ha='right' if xlabel_rotation > 0 else 'center',
                                               fontsize=8)
                    if ytick_size:
                        ax_len.tick_params(axis="y", labelsize=ytick_size)
                else:
                    ax_len.set_ylabel("")
                    ax_len.tick_params(axis="y", which="both", labelleft=False)
            else:
                # Parameter on y-axis, cell lengths on x-axis
                if ts_idx == 0:
                    ax_len.set_ylabel(f"{p_xlabel}{p_unit}")
                    if not is_num:
                        ax_len.set_yticks(param_vals)
                        ax_len.set_yticklabels(p_val_keys, fontsize=8)
                    if ytick_size:
                        ax_len.tick_params(axis="y", labelsize=ytick_size)
                else:
                    ax_len.set_ylabel("")
                    ax_len.tick_params(axis="y", which="both", labelleft=False)

            # x-axis label per column (bottom row only)
            if i == n_params - 1:
                if swap_left_xy:
                    # x-axis shows parameter values
                    if not is_num:
                        ax_len.set_xticks(param_vals)
                        ax_len.set_xticklabels(p_val_keys, rotation=xlabel_rotation,
                                               ha='right' if xlabel_rotation > 0 else 'center',
                                               fontsize=8)
                    # Set xlabel for the parameter
                    if left_xlabels is not None:
                        ax_len.set_xlabel(left_xlabels[ts_idx])
                else:
                    # x-axis shows cell lengths
                    xlabel = (
                        left_xlabels[ts_idx]
                        if left_xlabels is not None
                        else "(" + ", ".join(cell_tag_sets[ts_idx]) + ")"
                    )
                    ax_len.set_xlabel(xlabel)
            else:
                ax_len.set_xticklabels([])

            # example A/B horizontal/vertical bars
            if ts_idx == 0:
                for j, ev in enumerate(example_vals):
                    if show_AB:
                        if swap_left_xy:
                            ax_len.axvline(ev, color="k", lw=0.5, ls="-", alpha=0.5)
                            ax_len.text(
                                ev,
                                ax_len.get_ylim()[1] * 0.95,
                                AB_symbols[j],
                                ha="center",
                                va="top",
                                fontsize=10,
                                path_effects=[
                                    path_effects.withStroke(linewidth=3, foreground="white")
                                ],
                            )
                        else:
                            ax_len.axhline(ev, color="k", lw=0.5, ls="-", alpha=0.5)
                            ax_len.text(
                                AB_x[i] if i < len(AB_x) else AB_x[-1],
                                ev,
                                AB_symbols[j],
                                ha="center",
                                va="center",
                                fontsize=12,
                                path_effects=[
                                    path_effects.withStroke(linewidth=3, foreground="white")
                                ],
                            )
            else:
                for ev in example_vals:
                    if swap_left_xy:
                        ax_len.axvline(ev, color="k", lw=0.5, ls="-", alpha=0.5)
                    else:
                        ax_len.axhline(ev, color="k", lw=0.5, ls="-", alpha=0.5)
            
            # Clean up spines
            ax_len.spines['top'].set_visible(False)
            ax_len.spines['right'].set_visible(False)

        # ========== MATCH-RATE column (optional) - URESULTS =========================
        if show_mid:
            ax_mid = axes_uresults[i, num_length_plots]
            mR, _, _ = compute_match_stats(uresults, p_key, p_val_keys, handle_empty=True)

            if swap_left_xy:
                ax_mid.scatter(
                    param_vals,
                    mR,
                    color=colors_picks["pred_u"],
                    marker="^",
                    s=20,
                    edgecolors=colors_picks["pred_u"],
                    facecolors="white",
                    lw=1,
                    zorder=9,
                    alpha=0.75,
                )
                ax_mid.set_ylim(-0.1, 1.1)
                if i == n_params - 1:
                    ax_mid.set_xlabel("MR")
                else:
                    ax_mid.set_xticklabels([])
            else:
                ax_mid.scatter(
                    mR,
                    param_vals,
                    color=colors_picks["pred_u"],
                    marker="^",
                    s=20,
                    edgecolors=colors_picks["pred_u"],
                    facecolors="white",
                    lw=1,
                    zorder=9,
                    alpha=0.75,
                )
                ax_mid.set_xlim(-0.1, 1.1)
                if i == n_params - 1:
                    ax_mid.set_xlabel("MR")
                else:
                    ax_mid.set_xticklabels([])
            
            ax_mid.set_yticklabels([])
            ax_mid.grid(axis="x" if not swap_left_xy else "y", alpha=0.5)
            ax_mid.spines['top'].set_visible(False)
            ax_mid.spines['right'].set_visible(False)

    # ------------------------------------------------------------------ #
    # Create combined legend at the top
    # ------------------------------------------------------------------ #
    handles_combined = []
    labels_combined = []

    # Get handles from first row, first column of results
    h_res, l_res = axes_results[0, 0].get_legend_handles_labels()

    # Get handles from first row, first column of uresults
    h_ures, l_ures = axes_uresults[0, 0].get_legend_handles_labels()

    handles_combined.extend(h_res)
    handles_combined.extend(h_ures)
    
    handles_combined.append(
        Patch(
            facecolor=colors_picks["pred"],
            alpha=band_alpha,
            edgecolor=band_edgecolor,
            linewidth=band_lw,
            label=r"$\pm\sigma$ (deCIFer)",
        )
    )
    
    handles_combined.append(
        Patch(
            facecolor=colors_picks["pred_u"],
            alpha=band_alpha,
            edgecolor=band_edgecolor,
            linewidth=band_lw,
            label=r"$\pm\sigma$ (U-deCIFer)",
        )
    )

    labels_combined.extend(l_res)
    labels_combined.extend(l_ures)
    labels_combined.append(r"$\pm\sigma$ (deCIFer)")
    labels_combined.append(r"$\pm\sigma$ (U-deCIFer)")
    
    # Place left legend (cell parameters)
    axes_results[0, 0].legend(
        handles=handles_combined,
        labels=labels_combined,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=bbox_left,
        ncol=ncol_left,
        frameon=False,
        columnspacing=0.5,
        handletextpad=0.3,
    )
    
    # Get PXRD legend elements from right columns
    h_pxrd_res, l_pxrd_res = axes_results[0, -1].get_legend_handles_labels()
    h_pxrd_ures, l_pxrd_ures = axes_uresults[0, -1].get_legend_handles_labels()
    
    # Combine PXRD legends (remove duplicates)
    handles_pxrd = []
    labels_pxrd = []
    seen_labels = set()
    for h, l in zip(h_pxrd_res + h_pxrd_ures, l_pxrd_res + l_pxrd_ures):
        if l not in seen_labels:
            handles_pxrd.append(h)
            labels_pxrd.append(l)
            seen_labels.add(l)
    
    # Place right legend (PXRD)
    axes_results[0, -1].legend(
        handles=handles_pxrd,
        labels=labels_pxrd,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=bbox_right,
        ncol=ncol_right,
        frameon=False,
    )

    # ------------------------------------------------------------------ #
    # Harmonise x-limits across *all* length axes
    # ------------------------------------------------------------------ #
    all_length_axes = np.concatenate([
        axes_results[:, :num_length_plots].ravel(),
        axes_uresults[:, :num_length_plots].ravel()
    ])
    
    for idx, ax in enumerate(all_length_axes):
        if swap_left_xy:
            ylims = ax.get_ylim()
            ymin = cell_mins[idx % num_length_plots] if (cell_mins is not None and cell_mins[idx % num_length_plots] is not None) else ylims[0]
            ymax = cell_maxs[idx % num_length_plots] if (cell_maxs is not None and cell_maxs[idx % num_length_plots] is not None) else ylims[1]
            ax.set_ylim(ymin, ymax)
        else:
            xlims = ax.get_xlim()
            xmin = cell_mins[idx % num_length_plots] if (cell_mins is not None and cell_mins[idx % num_length_plots] is not None) else xlims[0]
            xmax = cell_maxs[idx % num_length_plots] if (cell_maxs is not None and cell_maxs[idx % num_length_plots] is not None) else xlims[1]
            ax.set_xlim(xmin, xmax)

    plt.subplots_adjust(hspace=0.2)

    if savepath:
        save_pdf_and_png(fig, savepath, dpi=dpi, facecolor="white")
    
    plt.close(fig)
    return fig

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b),
                                        map_location='cpu',
                                        weights_only=False)
        return super().find_class(module, name)

def cpu_load(path):
    with open(path, "rb") as f:
        return CPU_Unpickler(f).load()

# Crystal system hierarchy (lower index = higher symmetry)
SYMMETRY_HIERARCHY = {
    "cubic": 0,
    "hexagonal": 1,
    "trigonal": 2,
    "tetragonal": 3,
    "orthorhombic": 4,
    "monoclinic": 5,
    "triclinic": 6,
    None: 7,
}


def get_symmetry_relation(pred_system, ref_system):
    """
    Compare predicted vs reference crystal system.
    
    Returns:
        'match': predicted == reference
        'higher': predicted has higher symmetry (lower hierarchy index)
        'lower': predicted has lower symmetry (higher hierarchy index)
    """
    if pred_system is None or ref_system is None:
        return 'lower'  # Treat unknown as lower symmetry
    
    pred_rank = SYMMETRY_HIERARCHY.get(pred_system, 7)
    ref_rank = SYMMETRY_HIERARCHY.get(ref_system, 7)
    
    if pred_rank == ref_rank:
        return 'match'
    elif pred_rank < ref_rank:
        return 'higher'
    else:
        return 'lower'


# Marker mapping for symmetry relation
SYMMETRY_MARKERS = {
    'match': 'o',   # Circle - correct
    'higher': '^',  # Up triangle - higher symmetry than expected
    'lower': 'v',   # Down triangle - lower symmetry (degraded)
}


def plot_crystal_analysis(
    runs,
    param_keys,
    tag_set=("a", "b", "c"),
    savepath=None,
    dpi=150,
    figsize=(10, 5.5),
    
    # Color scheme - clean, accessible palette
    colors=None,
    
    # Typography
    title_fontsize=11,
    label_fontsize=10,
    tick_fontsize=9,
    legend_fontsize=9,
    
    # Y-axis labels
    ratio_ylabel=r"$\Delta\mathrm{RD}_{abc}$",
    deltamr_ylabel=r"$\Delta\mathrm{MR}$",
    
    # Visual styling
    marker_size=40,
    line_width=1.2,
    line_alpha=0.4,
    
    # Symmetry options
    show_symmetry=True,
    symprec=0.1,
    angle_tolerance=5,
    symmetry_threshold=0.60,
    
    # PXRD inset options
    show_pxrd=True,
    pxrd_examples_map=None,
    pxrd_xlim_map=None,

    # Optional confidence interval overlay
    ci_df=None,
    ci_band_alpha=0.18,
    
    # Label mapping
    run_label_map=None,
    
    # Helper functions (must be provided or defined)
    extract_numeric_property=None,
    
    # w/h space
    wspace_top=0.08,
    hspace_top=0.05,
    
    xlabel_rotation = 0,
):
    """
    Create a clean 2-row visualization of crystallographic analysis results.
    
    Marker shapes indicate symmetry prediction quality:
    - ○ (circle): Predicted crystal system matches reference
    - △ (up triangle): Predicted system has higher symmetry than reference
    - ▽ (down triangle): Predicted system has lower symmetry than reference
    
    Parameters
    ----------
    runs : list of tuples
        Each tuple is (label, deCIFer_results, U_results)
    param_keys : list of dicts
        Each dict has 'key', 'xlabel', 'unit', optionally 'xscale'
    """
    
    # ============================================================
    # IMPORTS & SETUP
    # ============================================================
    from pymatgen.core import Structure as PMGStructure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    
    # Default color palette - colorblind-friendly
    if colors is None:
        palette = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9']
        colors = {lab: palette[i % len(palette)] for i, (lab, _, _) in enumerate(runs)}

    # ============================================================
    # HELPER FUNCTIONS
    # ============================================================
    
    def _sort_key(k):
        try:
            return (0, float(k))
        except:
            return (1, str(k))

    def _condition_key(k):
        try:
            return str(float(k))
        except Exception:
            return str(k)
    
    def _get_prop(obj, tag):
        if extract_numeric_property and isinstance(obj, str):
            return extract_numeric_property(obj, tag)
        if isinstance(obj, PMGStructure):
            return getattr(obj.lattice, tag.split("_")[-1])
        raise TypeError("Unsupported structure type")
    
    def axis_vals(obj, dims):
        return [_get_prop(obj, f"_cell_length_{d}") for d in dims]
    
    def extract_ref_obj(entry):
        if not isinstance(entry, dict):
            return None
        be = entry.get("best_experiment")
        if isinstance(be, dict):
            for key in ("reference_structure", "generated_cif"):
                if key in be:
                    return be[key]
        exps = entry.get("experiments")
        if isinstance(exps, (list, tuple)):
            for e in exps:
                if isinstance(e, dict):
                    for key in ("reference_structure", "generated_cif"):
                        if key in e:
                            return e[key]
        return None
    
    def rel_spread_axes_mean(store, p_key, k):
        bucket = store.get(p_key)
        if not isinstance(bucket, dict):
            return np.nan
        entry = bucket.get(k)
        if not isinstance(entry, dict):
            return np.nan
        
        ref_obj = extract_ref_obj(entry)
        if ref_obj is None:
            return np.nan
        try:
            ref_mean = np.mean(axis_vals(ref_obj, tag_set))
        except:
            return np.nan
        if np.isnan(ref_mean) or ref_mean == 0:
            return np.nan
        
        exps = entry.get("experiments", [])
        if not exps:
            return np.nan
        
        vals_per_axis = {d: [] for d in tag_set}
        for e in exps:
            obj = e.get("generated_cif")
            if obj is None:
                continue
            try:
                av = axis_vals(obj, tag_set)
                for d, v in zip(tag_set, av):
                    vals_per_axis[d].append(v)
            except:
                continue
        
        sigmas = []
        for d in tag_set:
            arr = np.array(vals_per_axis[d], dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size:
                sigmas.append(np.std(arr))
        if not sigmas:
            return np.nan
        return np.mean(sigmas) / ref_mean
    
    def mean_mr(store, p_key, k):
        exps = store.get(p_key, {}).get(k, {}).get("experiments", [])
        arr = np.array([e.get("structure_match", np.nan) for e in exps], float)
        arr = arr[~np.isnan(arr)]
        return arr.mean() if arr.size else np.nan
    
    def extract_pxrd(entry):
        if not entry:
            return None
        be = entry.get("best_experiment", {})
        ref = be.get("pxrd_ref")
        gen = be.get("pxrd_gen_clean")
        if ref is None or gen is None:
            return None
        try:
            return (ref["q"], ref["iq"], gen["q_disc"][0], gen["iq_disc"][0])
        except:
            return None
    
    # Symmetry analysis caches
    _struct_cache = {}
    _sym_cache = {}
    
    def _to_structure(obj):
        if isinstance(obj, PMGStructure):
            return obj
        if isinstance(obj, str):
            if obj in _struct_cache:
                return _struct_cache[obj]
            try:
                st = PMGStructure.from_str(obj, fmt="cif")
            except:
                st = None
            _struct_cache[obj] = st
            return st
        return None
    
    def infer_symmetry(struct):
        if struct is None:
            return (None, None, None)
        key = id(struct)
        if key in _sym_cache:
            return _sym_cache[key]
        try:
            sga = SpacegroupAnalyzer(struct, symprec=symprec, angle_tolerance=angle_tolerance)
            out = (sga.get_crystal_system(), sga.get_space_group_symbol(), sga.get_space_group_number())
        except:
            out = (None, None, None)
        _sym_cache[key] = out
        return out
    
    def get_symmetry_info(store, p_key, k):
        """
        Returns (ref_system, top_predicted_system, relation)
        where relation is 'match', 'higher', or 'lower'
        """
        entry = store.get(p_key, {}).get(k, {})
        if not isinstance(entry, dict):
            return (None, None, 'lower')
        
        # Get reference crystal system
        ref_obj = extract_ref_obj(entry)
        ref_st = _to_structure(ref_obj) if show_symmetry else None
        ref_cs, _, _ = infer_symmetry(ref_st)
        
        exps = entry.get("experiments", [])
        if not exps:
            return (ref_cs, None, 'lower')
        
        # Count predicted crystal systems
        cs_counter = Counter()
        n_ok = 0
        for e in exps:
            st = _to_structure(e.get("generated_cif"))
            cs_i, _, _ = infer_symmetry(st)
            if cs_i:
                n_ok += 1
                cs_counter[cs_i] += 1
        
        if n_ok == 0:
            return (ref_cs, None, 'lower')
        
        # Get most common predicted system
        top_cs, top_n = cs_counter.most_common(1)[0]
        
        # Determine relation to reference
        relation = get_symmetry_relation(top_cs, ref_cs)
        
        return (ref_cs, top_cs, relation)
    
    # ============================================================
    # FIGURE SETUP
    # ============================================================
    
    n_cols = len(param_keys)
    
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='white')
    
    # Main grid for data rows
    gs_main = GridSpec(2, n_cols, figure=fig,
                       left=0.10, right=0.98, top=0.85, bottom=0.25,
                       hspace=hspace_top, wspace=wspace_top)

    # Separate grid for PXRD, positioned lower
    gs_pxrd = GridSpec(1, n_cols, figure=fig,
                       left=0.10, right=0.98, top=0.12, bottom=0.02,
                       wspace=0.08)

    axes_top = [fig.add_subplot(gs_main[0, i]) for i in range(n_cols)]
    axes_bot = [fig.add_subplot(gs_main[1, i]) for i in range(n_cols)]
    axes_pxrd = [fig.add_subplot(gs_pxrd[0, i]) for i in range(n_cols)] if show_pxrd else None
    
    # Track data for consistent y-limits
    all_y_top, all_y_bot = [], []
    run_handles, run_labels = [], []
    
    # ============================================================
    # PLOTTING
    # ============================================================
    
    for c, param in enumerate(param_keys):
        p_key = param["key"]
        xlabel = param["xlabel"]
        unit = param.get("unit", "")
        xscale = param.get("xscale", "linear")
        
        # Get union of all x-values
        union_keys = sorted(
            set().union(*(r[1].get(p_key, {}).keys() for r in runs)),
            key=_sort_key
        )
        
        # Determine if numeric
        try:
            xvals = np.array([float(k) for k in union_keys])
            numeric = True
        except:
            xvals = np.arange(len(union_keys))
            numeric = False
        
        ax_top, ax_bot = axes_top[c], axes_bot[c]
        
        # Clean up axes
        for ax in [ax_top, ax_bot]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.axhline(0, color='#999999', lw=0.8, ls='--', zorder=0)
            ax.tick_params(labelsize=tick_fontsize)
        
        # Set log scale if needed
        if numeric and xscale == "log":
            if np.all(xvals > 0):
                ax_top.set_xscale("log")
                ax_bot.set_xscale("log")
        
        # Plot each run
        for lab, res, ures in runs:
            col = colors[lab]
            display_label = run_label_map.get(lab, lab) if run_label_map else lab
            
            # Compute ΔRD values
            y_d = np.array([rel_spread_axes_mean(res, p_key, k) for k in union_keys], float)
            if ures is None:
                y_top = np.full_like(y_d, np.nan)
            else:
                y_u = np.array([rel_spread_axes_mean(ures, p_key, k) for k in union_keys], float)
                y_top = np.where((~np.isnan(y_d)) & (~np.isnan(y_u)), y_d - y_u, np.nan)
            
            # Compute ΔMR values
            if ures is None:
                y_bot = np.full(len(union_keys), np.nan)
            else:
                y_bot = np.array([mean_mr(res, p_key, k) - mean_mr(ures, p_key, k) 
                                  for k in union_keys], float)

            if ci_df is not None:
                ci_run_df = ci_df[
                    (ci_df["parameter_key"] == p_key)
                    & (ci_df["run_label"] == display_label)
                ].copy()
                if not ci_run_df.empty:
                    ci_run_df["__cond_key"] = ci_run_df["condition_key"].map(_condition_key)
                    rd_lookup = dict(zip(ci_run_df["__cond_key"], ci_run_df["delta_rdabc"]))
                    mr_lookup = dict(zip(ci_run_df["__cond_key"], ci_run_df["delta_mr"]))
                    ci_y_top = np.array([rd_lookup.get(_condition_key(k), np.nan) for k in union_keys], float)
                    ci_y_bot = np.array([mr_lookup.get(_condition_key(k), np.nan) for k in union_keys], float)
                    y_top = np.where(np.isfinite(ci_y_top), ci_y_top, y_top)
                    y_bot = np.where(np.isfinite(ci_y_bot), ci_y_bot, y_bot)
            
            # Masks for valid data
            m_top = ~np.isnan(y_top)
            m_bot = ~np.isnan(y_bot)
            
            # Get symmetry relations for marker shapes
            if show_symmetry:
                markers = []
                for k in union_keys:
                    ref_cs, top_cs, relation = get_symmetry_info(res, p_key, k)
                    markers.append(SYMMETRY_MARKERS[relation])
                markers = np.array(markers, dtype=object)
            
            # Plot connecting line (subtle)
            if np.sum(m_top) > 1:
                if ci_df is not None:
                    ci_run_df = ci_df[
                        (ci_df["parameter_key"] == p_key)
                        & (ci_df["run_label"] == display_label)
                    ].copy()
                    if not ci_run_df.empty:
                        ci_run_df["__cond_key"] = ci_run_df["condition_key"].map(_condition_key)
                        x_lookup = {_condition_key(key): (float(key) if numeric else union_keys.index(key)) for key in union_keys}
                        ci_run_df = ci_run_df[ci_run_df["__cond_key"].isin(x_lookup)]
                        ci_run_df["__sort_x"] = ci_run_df["__cond_key"].map(x_lookup)
                        ci_run_df = ci_run_df.sort_values("__sort_x")
                        ci_x = ci_run_df["__sort_x"].to_numpy(dtype=float)
                        lo_rd = ci_run_df["delta_rdabc_ci_low"].to_numpy(dtype=float)
                        hi_rd = ci_run_df["delta_rdabc_ci_high"].to_numpy(dtype=float)
                        finite = np.isfinite(ci_x) & np.isfinite(lo_rd) & np.isfinite(hi_rd)
                        if np.any(finite):
                            ax_top.fill_between(
                                ci_x[finite],
                                lo_rd[finite],
                                hi_rd[finite],
                                color=col,
                                alpha=ci_band_alpha,
                                linewidth=0,
                                zorder=0.5,
                            )
                ax_top.plot(xvals[m_top], y_top[m_top], color=col, lw=line_width, 
                           alpha=line_alpha, zorder=1)
            if np.sum(m_bot) > 1:
                if ci_df is not None:
                    ci_run_df = ci_df[
                        (ci_df["parameter_key"] == p_key)
                        & (ci_df["run_label"] == display_label)
                    ].copy()
                    if not ci_run_df.empty:
                        ci_run_df["__cond_key"] = ci_run_df["condition_key"].map(_condition_key)
                        x_lookup = {_condition_key(key): (float(key) if numeric else union_keys.index(key)) for key in union_keys}
                        ci_run_df = ci_run_df[ci_run_df["__cond_key"].isin(x_lookup)]
                        ci_run_df["__sort_x"] = ci_run_df["__cond_key"].map(x_lookup)
                        ci_run_df = ci_run_df.sort_values("__sort_x")
                        ci_x = ci_run_df["__sort_x"].to_numpy(dtype=float)
                        lo_mr = ci_run_df["delta_mr_ci_low"].to_numpy(dtype=float)
                        hi_mr = ci_run_df["delta_mr_ci_high"].to_numpy(dtype=float)
                        finite = np.isfinite(ci_x) & np.isfinite(lo_mr) & np.isfinite(hi_mr)
                        if np.any(finite):
                            ax_bot.fill_between(
                                ci_x[finite],
                                lo_mr[finite],
                                hi_mr[finite],
                                color=col,
                                alpha=ci_band_alpha,
                                linewidth=0,
                                zorder=0.5,
                            )
                ax_bot.plot(xvals[m_bot], y_bot[m_bot], color=col, lw=line_width,
                           alpha=line_alpha, zorder=1)
            
            # Plot markers
            if show_symmetry:
                # Plot each point with its appropriate marker
                for i, (x, y_t, y_b, valid_t, valid_b) in enumerate(zip(xvals, y_top, y_bot, m_top, m_bot)):
                    mk = markers[i]
                    if valid_t:
                        ax_top.scatter([x], [y_t], s=marker_size, marker=mk,
                                      edgecolors=col, facecolors=col,
                                      linewidths=1.0, zorder=3)
                        all_y_top.append(y_t)
                    if valid_b:
                        ax_bot.scatter([x], [y_b], s=marker_size, marker=mk,
                                      edgecolors=col, facecolors=col,
                                      linewidths=1.0, zorder=3)
                        all_y_bot.append(y_b)
            else:
                if np.any(m_top):
                    ax_top.scatter(xvals[m_top], y_top[m_top], s=marker_size,
                                  color=col, zorder=3)
                    all_y_top.extend(y_top[m_top])
                if np.any(m_bot):
                    ax_bot.scatter(xvals[m_bot], y_bot[m_bot], s=marker_size,
                                  color=col, zorder=3)
                    all_y_bot.extend(y_bot[m_bot])
            
            # Collect legend handles (only once per run)
            if c == 0 and lab not in [l for l in run_labels]:
                handle = mlines.Line2D([], [], color=col, marker='o', 
                                       markersize=7, linestyle='-', lw=line_width)
                run_handles.append(handle)
                run_labels.append(display_label)
        
        # X-axis labels
        ax_top.set_xticklabels([])
        
        if numeric:
            ax_bot.set_xlabel(f"{xlabel}" + (f" ({unit})" if unit else ""), 
                             fontsize=label_fontsize)
            
            # Apply rotation if specified (for numeric but discrete-like data)
            if xlabel_rotation != 0:
                ax_bot.tick_params(axis='x', rotation=xlabel_rotation)
                # Adjust alignment for rotated labels
                for tick in ax_bot.get_xticklabels():
                    tick.set_ha('right' if xlabel_rotation > 0 else 'left')
        else:
            ax_bot.set_xticks(xvals)
            ax_bot.set_xticklabels(union_keys, fontsize=tick_fontsize, 
                                   rotation=xlabel_rotation,
                                   ha='right' if xlabel_rotation > 0 else 'center')
            ax_bot.set_xlabel(xlabel + (f" ({unit})" if unit else ""), 
                             fontsize=label_fontsize)
        
        # Y-axis labels (only first column)
        if c == 0:
            ax_top.set_ylabel(ratio_ylabel, fontsize=label_fontsize)
            ax_bot.set_ylabel(deltamr_ylabel, fontsize=label_fontsize)
        else:
            ax_top.set_yticklabels([])
            ax_bot.set_yticklabels([])
        
        # ============================================================
        # PXRD INSETS
        # ============================================================
        if show_pxrd and axes_pxrd:
            ax_pxrd = axes_pxrd[c]
            
            # Get example keys
            if pxrd_examples_map and p_key in pxrd_examples_map:
                k_min, k_max = pxrd_examples_map[p_key]
            elif len(union_keys) >= 2:
                k_min, k_max = union_keys[0], union_keys[-1]
            else:
                k_min, k_max = None, None
            
            first_store = runs[0][1]
            
            if k_min and k_max:
                px_min = extract_pxrd(first_store.get(p_key, {}).get(k_min, {}))
                px_max = extract_pxrd(first_store.get(p_key, {}).get(k_max, {}))
                
                if px_min:
                    ax_pxrd.plot(px_min[0], px_min[1], color='#333333', lw=0.8, alpha=0.9)
                if px_max:
                    ax_pxrd.plot(px_max[0], px_max[1], color='#333333', lw=0.8, 
                                alpha=0.9, ls='--')

                if pxrd_xlim_map and p_key in pxrd_xlim_map:
                    ax_pxrd.set_xlim(pxrd_xlim_map[p_key])
            
            ax_pxrd.text(0.5, -0.12, r'$Q\;[Å^{-1}]$', transform=ax_pxrd.transAxes,
                           ha='center', va='top', fontsize=7, clip_on=False)
            ax_pxrd.patch.set_alpha(0.0)
            ax_pxrd.axis("off")
    
    # ============================================================
    # CONSISTENT Y-LIMITS
    # ============================================================
    
    def compute_ylim(values, padding=0.1):
        arr = np.array(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return (-0.5, 0.5)
        vmin, vmax = arr.min(), arr.max()
        margin = (vmax - vmin) * padding if vmax > vmin else 0.1
        return (vmin - margin, vmax + margin)
    
    ylim_top = compute_ylim(all_y_top)
    ylim_bot = compute_ylim(all_y_bot)
    
    for ax in axes_top:
        ax.set_ylim(ylim_top)
    for ax in axes_bot:
        ax.set_ylim(ylim_bot)
    
    # ============================================================
    # LEGENDS
    # ============================================================
    
    # Main legend for runs (polymorphs)
    fig.legend(run_handles, run_labels, 
              loc='upper center', 
              bbox_to_anchor=(0.54, 0.99),
              ncol=len(run_labels),
              frameon=False, 
              fontsize=legend_fontsize,
              handletextpad=0.5,
              columnspacing=1.5)
    
    # Symmetry legend (simplified: match/higher/lower)
    if show_symmetry:
        match_handle = mlines.Line2D([], [], marker='o', linestyle='None',
                                     markerfacecolor='gray', markeredgecolor='gray',
                                     markersize=6)
        higher_handle = mlines.Line2D([], [], marker='^', linestyle='None',
                                      markerfacecolor='gray', markeredgecolor='gray',
                                      markersize=6)
        lower_handle = mlines.Line2D([], [], marker='v', linestyle='None',
                                     markerfacecolor='gray', markeredgecolor='gray',
                                     markersize=6)
        
        fig.legend([match_handle, higher_handle, lower_handle], 
                  ['pred. sym. = ref.', 'pred. sym. > ref.','pred. sym. < ref.'],
                  loc='upper center',
                  bbox_to_anchor=(0.54, 0.93),
                  ncol=3,
                  frameon=False,
                  fontsize=legend_fontsize - 1,
                  title_fontsize=legend_fontsize - 1,
                  handletextpad=0.3,
                  columnspacing=1.0)
    
    # ============================================================
    # SAVE & SHOW
    # ============================================================
    
    if savepath:
        save_pdf_and_png(fig, savepath, dpi=dpi, facecolor="white")
    
    plt.close(fig)

def plot_unit_cell_with_boundaries(
    structure: Structure,
    ax: Optional[plt.Axes] = None,
    tol: float = 1e-5,
    radii: float = 0.8,
    rotation: Union[str, Tuple[str, ...]] = ('45x, -15y, 90z'),
    offset: Tuple[float, float, float] = (0, 0, 0),
    scale: float = 1.0,
) -> Tuple[plt.Axes, PMGStructure]:
    """
    Plots the unit cell with periodic boundaries using ASE and pymatgen.

    Parameters:
        structure (Structure): The pymatgen Structure to plot.
        ax (Optional[plt.Axes]): Matplotlib axes to plot on. If None, a new figure and axes are created.
        tol (float): Tolerance for duplicate fractional coordinates.
        radii (float): Radius for the atoms in the plot.
        rotation (Union[str, Tuple[str, ...]]): Rotation applied to the structure.
        offset (Tuple[float, float, float]): Offset applied to the structure.

    Returns:
        Tuple[plt.Axes, PMGStructure]: The axes and the discrete structure created for plotting.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Define translation vectors for periodic images
    translation_vectors = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]
    ]

    all_species: List[str] = []
    all_coords: List[np.ndarray] = []

    # Loop over each translation vector to create periodic images
    for tv in translation_vectors:
        tv_cart = structure.lattice.get_cartesian_coords(tv)
        for site in structure:
            if tv == [0, 0, 0]:
                all_species.append(site.species_string)
                all_coords.append(site.coords)
            else:
                # Only add sites that satisfy the fractional coordinate condition
                if all(site.frac_coords[i] < tol for i, shift in enumerate(tv) if shift == 1):
                    all_species.append(site.species_string)
                    all_coords.append(site.coords + tv_cart)

    # Create a discrete structure using pymatgen
    discrete_structure = PMGStructure(
        lattice=structure.lattice.matrix,
        species=all_species,
        coords=np.array(all_coords),
        coords_are_cartesian=True
    )

    # Convert the structure to ASE atoms and disable periodic boundary conditions for plotting
    ase_atoms = AseAtomsAdaptor.get_atoms(discrete_structure)
    ase_atoms.set_pbc([False, False, False])
    plot_atoms(ase_atoms, ax, radii=radii, show_unit_cell=True, rotation=rotation, offset=offset, scale=scale)

    return ax, discrete_structure

# Clean style matching the analysis plots
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.facecolor": "white",
})

# Colorblind-friendly palette (matching main figures)
COLORS = {
    "cubic": "#0072B2",      # Blue
    "hexagonal": "#E69F00",  # Orange
    "trigonal": "#009E73",   # Green
}


def pad_uc(ax, frac=0.15):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = x1 - x0
    dy = y1 - y0
    ax.set_xlim(x0 - frac*dx, x1 + frac*dx)
    ax.set_ylim(y0 - frac*dy, y1 + frac*dy)


def overlay_unit_cell(fig, rect, struct, scale, radii, rotation):
    """Pure overlay axes, fully independent, resizable."""
    ax_uc = fig.add_axes(rect, frameon=False)
    ax_uc.set_xticks([])
    ax_uc.set_yticks([])
    ax_uc.patch.set_alpha(0)
    plot_unit_cell_with_boundaries(
        struct,
        ax=ax_uc,
        offset=(0, 0, 0),
        scale=scale,
        radii=radii,
        rotation=rotation,
    )
    return ax_uc


def plot_feo2_polymorphs(
    # PXRD data
    q_cub, iq_cub, qd_cub,
    q_hex, iq_hex, qd_hex,
    q_tri, iq_tri, qd_tri,
    # Structures for unit cell visualization
    st_cub, st_hex, st_tri,
    # Output
    savepath="feo2_polymorphs.pdf",
    dpi=300,
    figsize=(6.73, 3.5),
):
    """
    Plot PXRD patterns and unit cells for three FeO2 polymorphs.

    Laid out at true double-column print size (17.1 cm wide): one column per
    polymorph with a large unit-cell rendering on top and the PXRD pattern
    below, so all font sizes here are the printed sizes.
    """
    from ase.data import atomic_numbers as _atomic_numbers, colors as _ase_colors

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    gs = GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.55, 1.0],
        left=0.035, right=0.995, top=0.875, bottom=0.125,
        wspace=0.12, hspace=0.10,
    )

    polymorphs = [
        {
            "title": r"FeO$_2$ cubic ($F\overline{4}3m$)",
            "color": COLORS["cubic"],
            "pxrd": (q_cub, iq_cub, qd_cub),
            "struct": st_cub,
            "rotation": "45x, -12.5y, 90z",
            "pad": 0.06,
        },
        {
            "title": r"FeO$_2$ hexagonal ($P6_3mc$)",
            "color": COLORS["hexagonal"],
            "pxrd": (q_hex, iq_hex, qd_hex),
            "struct": st_hex,
            "rotation": "45x, -12.5y, 90z",
            "pad": 0.06,
        },
        {
            "title": r"FeO$_2$ trigonal ($R3m$)",
            "color": COLORS["trigonal"],
            "pxrd": (q_tri, iq_tri, qd_tri),
            "struct": st_tri,
            "rotation": "20x, -50y, 0z",
            "pad": 0.08,
        },
    ]

    UC_RADII = 0.40

    for col, poly in enumerate(polymorphs):
        # --- Unit cell (top, large) ---
        ax_uc = fig.add_subplot(gs[0, col], frameon=False)
        ax_uc.set_xticks([])
        ax_uc.set_yticks([])
        ax_uc.patch.set_alpha(0)
        plot_unit_cell_with_boundaries(
            poly["struct"],
            ax=ax_uc,
            radii=UC_RADII,
            rotation=poly["rotation"],
        )
        pad_uc(ax_uc, frac=poly["pad"])
        ax_uc.set_aspect("equal", adjustable="box")
        # plot_atoms fits the limits tightly, so atoms on the hull get shaved
        # at the axes edge; let the artists draw past it instead.
        for artist in (
            list(ax_uc.lines) + list(ax_uc.patches)
            + list(ax_uc.collections) + list(ax_uc.artists)
        ):
            artist.set_clip_on(False)
        # Title at a uniform height per column (the equal-aspect box may
        # shrink, so anchor to the gridspec slot instead of the axes).
        slot = gs[0, col].get_position(fig)
        fig.text(
            (slot.x0 + slot.x1) / 2, slot.y1 + 0.025,
            poly["title"],
            ha="center", va="bottom", fontsize=9, color=poly["color"],
        )

        # --- PXRD (bottom) ---
        ax_px = fig.add_subplot(gs[1, col])
        q, iq, qd = poly["pxrd"]
        ax_px.plot(q, iq, lw=0.9, color=poly["color"])
        ax_px.bar(qd, height=-0.1, width=0.02, color=poly["color"])
        ax_px.set_ylim(-0.2, 1.25)
        ax_px.set_xlim(1.75, 7.25)
        ax_px.set_yticks([])
        ax_px.spines["top"].set_visible(False)
        ax_px.spines["right"].set_visible(False)
        ax_px.spines["left"].set_visible(False)
        ax_px.tick_params(labelsize=8)
        ax_px.set_xlabel(r"$Q$ [Å$^{-1}$]", fontsize=9)
        if col == 0:
            ax_px.set_ylabel(r"$I(Q)$ [a.u.]", fontsize=9)

        # --- Atom key (Fe / O) inside the first PXRD panel ---
        if col == 0:
            handles = [
                mlines.Line2D(
                    [0], [0], linestyle="None", marker="o", markersize=6,
                    markerfacecolor=_ase_colors.jmol_colors[_atomic_numbers[sp]],
                    markeredgecolor="black", markeredgewidth=0.6, label=sp,
                )
                for sp in ("Fe", "O")
            ]
            ax_px.legend(
                handles=handles, loc="upper right", frameon=False,
                fontsize=8, handletextpad=0.1, borderaxespad=0.2,
                labelspacing=0.3,
            )

    if savepath:
        save_pdf_and_png(
            fig,
            savepath,
            dpi=dpi,
            facecolor="white",
            pad_inches=0.02,
        )
    plt.close(fig)


def load_main_ablation_datasets():
    return {
        "results_feo2": cpu_load("pkl-files/ablation_mainfig_cubic_large_FeO_wider.pkl")["results"],
        "uresults_feo2": cpu_load("pkl-files/ablation_mainfig_cubic_large_nocond_FeO_wider.pkl")["results"],
        "results_feo2_hex": cpu_load("pkl-files/ablation_mainfig_hexagonal_large_FeO.pkl")["results"],
        "uresults_feo2_hex": cpu_load("pkl-files/ablation_mainfig_hexagonal_large_nocond_FeO.pkl")["results"],
        "results_feo2_tri": cpu_load("pkl-files/ablation_mainfig_trigonal_large_FeO.pkl")["results"],
        "uresults_feo2_tri": cpu_load("pkl-files/ablation_mainfig_trigonal_large_nocond_FeO.pkl")["results"],
    }


def load_appendix_ablation_datasets():
    return {
        "results_feo2_app_cub": cpu_load("pkl-files/ablation_appendix_cubic_large_FeO.pkl")["results"],
        "uresults_feo2_app_cub": cpu_load("pkl-files/ablation_appendix_cubic_large_nocond_FeO.pkl")["results"],
        "results_feo2_app_hex": cpu_load("pkl-files/ablation_appendix_hexagonal_large_FeO.pkl")["results"],
        "uresults_feo2_app_hex": cpu_load("pkl-files/ablation_appendix_hexagonal_large_nocond_FeO.pkl")["results"],
        "results_feo2_app_tri": cpu_load("pkl-files/ablation_appendix_trigonal_large_FeO.pkl")["results"],
        "uresults_feo2_app_tri": cpu_load("pkl-files/ablation_appendix_trigonal_large_nocond_FeO.pkl")["results"],
    }


def load_polymorph_datasets():
    return {
        "results_feo2_cub": cpu_load("pkl-files/ablation_mainfig_cubic_large_FeO.pkl")["results"],
        "results_feo2_hex": cpu_load("pkl-files/ablation_mainfig_hexagonal_large_FeO.pkl")["results"],
        "results_feo2_tri": cpu_load("pkl-files/ablation_mainfig_trigonal_large_FeO.pkl")["results"],
    }


def run_main_ablation_figures(output_dir: Path, datasets: dict):
    globals().update(datasets)
    output_dir.mkdir(parents=True, exist_ok=True)
    mpl.rcParams['hatch.linewidth'] = 0.5

    # ============================================================
    # POLYMORPH DATA
    # ============================================================

    polymorphs = [
        {
            "name": "cubic",
            "suffix": "cub",
            "results": results_feo2,
            "uresults": uresults_feo2,
        },
        {
            "name": "hexagonal", 
            "suffix": "hex",
            "results": results_feo2_hex,
            "uresults": uresults_feo2_hex,
        },
        {
            "name": "trigonal",
            "suffix": "tri",
            "results": results_feo2_tri,
            "uresults": uresults_feo2_tri,
        },
    ]

    # ============================================================
    # PARAMETER DEFINITIONS (Appendix distortions)
    # ============================================================

    # Each entry defines the distortion type and its display properties
    param_configs = [
        {
            "key": "q_pre_scale_uniform", 
            "xlabel": "$\\langle a,b,c \\rangle_{\\text{pred}}$ [Å]",
            "left_xlabel": "Unit cell scale factor",
            "example_idxs": [0,2,8,12,14],
            "example_idxs_uresults": [0,2,8,12,14],
            "discrete": False,
            "pxrd_annot": "scale",
        },
        {
            "key": "peak_asymmetry",
            "xlabel": r"$\langle a,b,c \rangle_{\text{pred}}$ [Å]",
            "left_xlabel": "Peak asymmetry",
            "example_idxs": [0, 2, 4, 6, 8],
            "example_idxs_uresults": [0, 2, 4, 6, 8],
            "discrete": False,
            "pxrd_annot": "asym",
        },
        {
            "key": "particle_size",
            "xlabel": r"$\langle a,b,c \rangle_{\text{pred}}$ [Å]",
            "left_xlabel": r"Crystallite size $\tau$ [Å]",
            "example_idxs": [0, 2, 4, 6, 8],
            "example_idxs_uresults": [0, 2, 4, 6, 8],
            "discrete": False,
            "pxrd_annot": r"$\tau$",
        },
    ]

    # ============================================================
    # GENERATE PLOTS
    # ============================================================

    for poly in polymorphs:
        poly_name = poly["name"]
        poly_suffix = poly["suffix"]
        res = poly["results"]
        ures = poly["uresults"]
    
        print(f"\n{'='*50}")
        print(f"Processing: {poly_name}")
        print(f"{'='*50}")
    
        for config in param_configs:
            p_key = config["key"]
            is_discrete = config.get("discrete", False)
            xlabel_rot = config.get("xlabel_rotation", 0)
        
            print(f"  Generating: {p_key}...")
        
            # Build param_keys for this single distortion
            param_keys = [{
                "key": p_key,
                "title": "",
                "xlabel": config["xlabel"],
                "unit": "",
                "example_idxs": config["example_idxs"],
            }]
        
            # Call ablation_fig with updated parameters
            ablation_fig(
                res,
                ures,
                param_keys,
                savepath=str(output_dir / f"FeO2_{p_key}_{poly_suffix}.pdf"),
                dpi=300,
                figscale=1.5,
                figlen=4.0,
                fig_height_factor=1.5,
                cell_mins=[None],
                cell_maxs=[None],
                bbox_left=(0.5, 1.35),
                ncol_left=2,
                bbox_right=(0.5, 1.4),
                ncol_right=1,
                wspace_outer=0.05,
                hspace_outer=0.05,
                wspace_left=0.1,
                show_AB=False,
                cell_tag_sets=[["a", "b", "c"]],
                left_xlabels=[config["left_xlabel"]],
                example_idxs_uresults=config["example_idxs_uresults"],
                show_mean=True,
                show_best=True,
                title_left_pad=17.0,
                show_mid=False,
                show_u=True,
                mid_size=0.3,
                show_best_stem=True,
                show_boxplots=False,
                width_ratios=[1.0, 1.0],
                x_txt=6,
                pxrd_xlim=(1, 7.5),
                best_markersize=30,
                pred_ms=2.5,
                # NEW parameters for discrete handling
                swap_left_xy=True,  # Keep consistent orientation
                xlabel_rotation=xlabel_rot,
                pxrd_annotation_key=config.get("pxrd_annot", None),
                panel_labels=["(a)", "(b)", "(c)", "(d)"],
            )


def run_appendix_ablation_figures(output_dir: Path, datasets: dict):
    globals().update(datasets)
    output_dir.mkdir(parents=True, exist_ok=True)
    # ============================================================
    # POLYMORPH DATA
    # ============================================================

    polymorphs = [
        {
            "name": "cubic",
            "suffix": "cub",
            "results": results_feo2_app_cub,
            "uresults": uresults_feo2_app_cub,
        },
        {
            "name": "hexagonal", 
            "suffix": "hex",
            "results": results_feo2_app_hex,
            "uresults": uresults_feo2_app_hex,
        },
        {
            "name": "trigonal",
            "suffix": "tri",
            "results": results_feo2_app_tri,
            "uresults": uresults_feo2_app_tri,
        },
    ]

    # ============================================================
    # PARAMETER DEFINITIONS (Appendix distortions)
    # ============================================================

    # Each entry defines the distortion type and its display properties
    param_configs = [
        {
            "key": "noise",
            "xlabel": r"$\langle a,b,c \rangle_{\text{pred}}$ [Å]",
            "left_xlabel": "Additive Noise",
            "example_idxs": [0, 2, 4, 6, 8],
            "example_idxs_uresults": [0, 2, 4, 6, 8],
            "discrete": False,
            "pxrd_annot": "noise",  # Annotation key for PXRD panels
        },
        {
            "key": "q_shift",
            "xlabel": r"$\langle a,b,c \rangle_{\text{pred}}$ [Å]",
            "left_xlabel": r"Peak shift [Å$^{-1}$]",
            "example_idxs": [0, 2, 4, 6, 8],
            "example_idxs_uresults": [0, 2, 4, 6, 8],
            "discrete": False,
            "pxrd_annot": "shift",
        },
        {
            "key": "base_fwhm",
            "xlabel": r"$\langle a,b,c \rangle_{\text{pred}}$ [Å]",
            "left_xlabel": "FWHM",
            "example_idxs": [0, 2, 4, 6, 8],
            "example_idxs_uresults": [0, 2, 4, 6, 8],
            "discrete": False,
            "pxrd_annot": "FWHM",
        },
        {
            "key": "chebychev_norm_coeffs",
            "xlabel": r"$\langle a,b,c \rangle_{\text{pred}}$ [Å]",
            "left_xlabel": "Background",
            "example_idxs": [0, 1, 2, 3, 4],
            "example_idxs_uresults": [0, 1, 2, 3, 4],
            "discrete": True,
            "pxrd_annot": "bg",
            "xlabel_rotation": 45,
        },
        {
            "key": "preferred_orientation_range",
            "xlabel": r"$\langle a,b,c \rangle_{\text{pred}}$ [Å]",
            "left_xlabel": "Pref. orient.",
            "example_idxs": [0, 2, 4, 6, 8],
            "example_idxs_uresults": [0, 2, 4, 6, 8],
            "discrete": True,
            "pxrd_annot": "orient",
            "xlabel_rotation": 45,
        },
        {
            "key": "mask_ranges",
            "xlabel": r"$\langle a,b,c \rangle_{\text{pred}}$ [Å]",
            "left_xlabel": "Q-mask",
            "example_idxs": [0, 2, 4, 6, 8],
            "example_idxs_uresults": [0, 2, 4, 6, 8],
            "discrete": True,
            "pxrd_annot": "mask",
            "xlabel_rotation": 45,
        },
    ]

    # ============================================================
    # GENERATE PLOTS
    # ============================================================

    for poly in polymorphs:
        poly_name = poly["name"]
        poly_suffix = poly["suffix"]
        res = poly["results"]
        ures = poly["uresults"]
    
        print(f"\n{'='*50}")
        print(f"Processing: {poly_name}")
        print(f"{'='*50}")
    
        for config in param_configs:
            p_key = config["key"]
            is_discrete = config.get("discrete", False)
            xlabel_rot = config.get("xlabel_rotation", 0)
        
            print(f"  Generating: {p_key}...")
        
            # Build param_keys for this single distortion
            param_keys = [{
                "key": p_key,
                "title": "",
                "xlabel": config["xlabel"],
                "unit": "",
                "example_idxs": config["example_idxs"],
            }]
        
            # Call ablation_fig with updated parameters
            ablation_fig(
                res,
                ures,
                param_keys,
                savepath=str(output_dir / f"FeO2_{p_key}_{poly_suffix}.pdf"),
                dpi=300,
                figscale=1.5,
                figlen=4.0,
                fig_height_factor=1.5,
                cell_mins=[None],
                cell_maxs=[None],
                bbox_left=(0.5, 1.35),
                ncol_left=2,
                bbox_right=(0.5, 1.4),
                ncol_right=1,
                wspace_outer=0.05,
                hspace_outer=0.05,
                wspace_left=0.1,
                show_AB=False,
                cell_tag_sets=[["a", "b", "c"]],
                left_xlabels=[config["left_xlabel"]],
                example_idxs_uresults=config["example_idxs_uresults"],
                show_mean=True,
                show_best=True,
                title_left_pad=17.0,
                show_mid=False,
                show_u=True,
                mid_size=0.3,
                show_best_stem=True,
                show_boxplots=False,
                width_ratios=[1.0, 1.0],
                x_txt=6,
                pxrd_xlim=(1, 7.5),
                best_markersize=30,
                pred_ms=2.5,
                # NEW parameters for discrete handling
                swap_left_xy=True,  # Keep consistent orientation
                xlabel_rotation=xlabel_rot,
                pxrd_annotation_key=config.get("pxrd_annot", None),
                panel_labels=["(a)", "(b)", "(c)", "(d)"],
            )


def run_main_robustness_figure(output_dir: Path, datasets: dict):
    globals().update(datasets)
    output_dir.mkdir(parents=True, exist_ok=True)
    param_keys = [
        {"key": "q_pre_scale_uniform", "xlabel": "unit cell scale factor", "unit": "", "xscale": "linear"},
        {"key": "peak_asymmetry",      "xlabel": "peak asymmetry",                   "unit": "", "xscale": "linear"},
        {"key": "particle_size",       "xlabel": r"crystallite size ($\tau$)",       "unit": "[Å]", "xscale": "log"},
    ]

    runs = [
        (r"FeO$_2$ (Cubic)",     results_feo2,     uresults_feo2),
        (r"FeO$_2$ (Hexagonal)", results_feo2_hex, uresults_feo2_hex),
        (r"FeO$_2$ (Trigonal)",  results_feo2_tri, uresults_feo2_tri),
    ]

    # PXRD inset configuration
    pxrd_examples_map = {
        "q_pre_scale_uniform": ("1.05", "1.0"),
        "peak_asymmetry":      ("0.1", "1.0"),
        "particle_size":       ("100", "10"),
    }

    pxrd_xlim_map = {
        "q_pre_scale_uniform": (2.0, 3.0),
        "peak_asymmetry":      (2.0, 3.0),
        "particle_size":       (2.0, 3.0),
    }

    # Color mapping for polymorphs
    colors = {
        r"FeO$_2$ (Cubic)":     "#0072B2",  # blue
        r"FeO$_2$ (Hexagonal)": "#E69F00",  # orange
        r"FeO$_2$ (Trigonal)":  "#009E73",  # green
    }
    ci_df = build_ci_summary_df_from_runs(
        runs,
        ["q_pre_scale_uniform", "peak_asymmetry", "particle_size"],
        tag_set=("a", "b", "c"),
        n_boot=10000,
        confidence=0.95,
        seed=0,
    )
    if ci_df is not None:
        ci_df = ci_df.loc[
            ~(
                (ci_df["parameter_key"] == "particle_size")
                & (ci_df["run_label"] == r"FeO$_2$ (Cubic)")
                & (ci_df["condition_key"].astype(str).isin(["1", "1.0"]))
            )
        ].copy()

    common_kwargs = dict(
        runs=runs,
        param_keys=param_keys,
        tag_set=("a", "b", "c"),
        figsize=tuple(np.array([6.75, 5.5])*0.8),
        dpi=150,
        colors=colors,
        legend_fontsize=8,
        label_fontsize=11,
        tick_fontsize=9,
        ratio_ylabel=r'$\Delta\mathrm{RD}_{abc}$',
        deltamr_ylabel=r'$\Delta\mathrm{MR}$',
        marker_size=16,
        line_width=1.2,
        line_alpha=0.5,
        show_symmetry=True,
        symprec=0.1,
        angle_tolerance=5,
        symmetry_threshold=0.60,
        show_pxrd=True,
        pxrd_examples_map=pxrd_examples_map,
        pxrd_xlim_map=pxrd_xlim_map,
        extract_numeric_property=extract_numeric_property,
        wspace_top=0.25,
        hspace_top=0.1,
    )

    # Original robustness figure without CI
    plot_crystal_analysis(
        savepath=str(output_dir / "FeO2_robustness.pdf"),
        ci_df=None,
        **common_kwargs,
    )

    # Updated robustness figure with CI
    plot_crystal_analysis(
        savepath=str(output_dir / "FeO2_robustness_ci.pdf"),
        ci_df=ci_df,
        **common_kwargs,
    )


def run_summary_robustness_figures(output_dir: Path, datasets: dict):
    globals().update(datasets)
    output_dir.mkdir(parents=True, exist_ok=True)
    """
    Execution code for clean crystallographic analysis plots.
    Generates SEPARATE plots for each distortion type with PXRD examples.

    - First 3 (noise, q_shift, base_fwhm): continuous x-axis with connecting lines
    - Last 3 (chebychev, orientation, masking): discrete/categorical, scatter only, angled labels
    """

    import numpy as np

    # ============================================================
    # COMMON SETTINGS
    # ============================================================

    runs = [
        (r"FeO$_2$ (Cubic)",     results_feo2_app_cub,     uresults_feo2_app_cub),
        (r"FeO$_2$ (Hexagonal)", results_feo2_app_hex,     uresults_feo2_app_hex),
        (r"FeO$_2$ (Trigonal)",  results_feo2_app_tri,     uresults_feo2_app_tri),
    ]

    colors = {
        r"FeO$_2$ (Cubic)":     "#0072B2",  # blue
        r"FeO$_2$ (Hexagonal)": "#E69F00",  # orange
        r"FeO$_2$ (Trigonal)":  "#009E73",  # green
    }
    ci_df = build_ci_summary_df_from_runs(
        runs,
        [
            "noise",
            "q_shift",
            "base_fwhm",
            "chebychev_norm_coeffs",
            "preferred_orientation_range",
            "mask_ranges",
        ],
        tag_set=("a", "b", "c"),
        n_boot=10000,
        confidence=0.95,
        seed=0,
    )

    # Common plot settings
    common_kwargs = dict(
        runs=runs,
        tag_set=("a", "b", "c"),
        figsize=(4.5, 4.5),  # Slightly taller to accommodate PXRD
        dpi=150,
        colors=colors,
        legend_fontsize=8,
        label_fontsize=10,
        tick_fontsize=9,
        ratio_ylabel=r'$\Delta\mathrm{RD}_{abc}$',
        deltamr_ylabel=r'$\Delta\mathrm{MR}$',
        marker_size=20,
        show_symmetry=True,
        symprec=0.1,
        angle_tolerance=5,
        symmetry_threshold=0.60,
        show_pxrd=False,
        ci_df=ci_df,
        extract_numeric_property=extract_numeric_property,
    )

    # Common Q range for PXRD insets
    default_pxrd_xlim = (1.5, 4.0)


    # ============================================================
    # 1. NOISE (continuous)
    # ============================================================
    plot_crystal_analysis(
        param_keys=[{
            "key": "noise", 
            "xlabel": "Additive Noise", 
            "unit": "", 
            "xscale": "linear"
        }],
        line_width=1.2,
        line_alpha=0.5,
        pxrd_examples_map={"noise": ("0.01", "0.15")},  # low vs high noise
        pxrd_xlim_map={"noise": default_pxrd_xlim},
        savepath=str(output_dir / "summary_noise.pdf"),
        **common_kwargs,
    )


    # ============================================================
    # 2. PEAK SHIFT (continuous)
    # ============================================================
    plot_crystal_analysis(
        param_keys=[{
            "key": "q_shift", 
            "xlabel": r"Peak position shift", 
            "unit": r"Å$^{-1}$", 
            "xscale": "linear"
        }],
        line_width=1.2,
        line_alpha=0.5,
        pxrd_examples_map={"q_shift": ("0.0", "0.1")},  # no shift vs large shift
        pxrd_xlim_map={"q_shift": default_pxrd_xlim},
        savepath=str(output_dir / "summary_shift.pdf"),
        **common_kwargs,
    )


    # ============================================================
    # 3. INSTRUMENTAL BROADENING (continuous)
    # ============================================================
    plot_crystal_analysis(
        param_keys=[{
            "key": "base_fwhm", 
            "xlabel": "Instrumental broadening (FWHM)", 
            "unit": "", 
            "xscale": "linear"
        }],
        line_width=1.2,
        line_alpha=0.5,
        pxrd_examples_map={"base_fwhm": ("0.01", "0.15")},  # sharp vs broad
        pxrd_xlim_map={"base_fwhm": default_pxrd_xlim},
        savepath=str(output_dir / "summary_fwhm.pdf"),
        **common_kwargs,
    )


    # ============================================================
    # 4. BACKGROUND / CHEBYCHEV COEFFS (discrete - scatter only)
    # ============================================================
    plot_crystal_analysis(
        param_keys=[{
            "key": "chebychev_norm_coeffs", 
            "xlabel": "Background (Chebyshev coeffs.)", 
            "unit": "", 
            "xscale": "linear"
        }],
        line_width=0,
        line_alpha=0,
        pxrd_examples_map={"chebychev_norm_coeffs": ("0", "5")},
        pxrd_xlim_map={"chebychev_norm_coeffs": default_pxrd_xlim},
        xlabel_rotation=45,
        savepath=str(output_dir / "summary_background.pdf"),
        **common_kwargs,
    )


    # ============================================================
    # 5. PREFERRED ORIENTATION (discrete - scatter only)
    # ============================================================
    plot_crystal_analysis(
        param_keys=[{
            "key": "preferred_orientation_range", 
            "xlabel": "Preferred orientation", 
            "unit": "", 
            "xscale": "linear"
        }],
        line_width=0,
        line_alpha=0,
        pxrd_examples_map={"preferred_orientation_range": ("0", "10")},
        pxrd_xlim_map={"preferred_orientation_range": default_pxrd_xlim},
        xlabel_rotation=45,
        savepath=str(output_dir / "summary_orientation.pdf"),
        **common_kwargs,
    )


    # ============================================================
    # 6. Q-MASKING (discrete - scatter only)
    # ============================================================
    plot_crystal_analysis(
        param_keys=[{
            "key": "mask_ranges", 
            "xlabel": "Q-masking ranges", 
            "unit": "", 
            "xscale": "linear"
        }],
        line_width=0,
        line_alpha=0,
        pxrd_examples_map={"mask_ranges": ("0", "10")},
        pxrd_xlim_map={"mask_ranges": default_pxrd_xlim},
        xlabel_rotation=45,
        savepath=str(output_dir / "summary_mask.pdf"),
        **common_kwargs,
    )



def run_polymorph_figure(output_dir: Path, datasets: dict):
    globals().update(datasets)
    output_dir.mkdir(parents=True, exist_ok=True)
    globals().update(load_polymorph_datasets())

    def extract(r):
        iq  = r['particle_size']['5']['best_experiment']["pxrd_ref_clean"]["iq"]
        q   = r['particle_size']['5']['best_experiment']["pxrd_ref_clean"]["q"]
        iqd = r['particle_size']['5']['best_experiment']["pxrd_ref_clean"]["iq_disc"][0]
        qd  = r['particle_size']['5']['best_experiment']["pxrd_ref_clean"]["q_disc"][0]
        st  = r['particle_size']['5']['best_experiment']["reference_structure"]
        return q, iq, qd, iqd, st

    q_cub, iq_cub, qd_cub, iqd_cub, st_cub = extract(results_feo2_cub)
    q_hex, iq_hex, qd_hex, iqd_hex, st_hex = extract(results_feo2_hex)
    q_tri, iq_tri, qd_tri, iqd_tri, st_tri = extract(results_feo2_tri)
    plot_feo2_polymorphs(
        q_cub, iq_cub, qd_cub,
        q_hex, iq_hex, qd_hex,
        q_tri, iq_tri, qd_tri,
        st_cub, st_hex, st_tri,
        savepath=str(output_dir / "feo2_polymorphs_v2.pdf"),
        figsize=(6.73, 3.5)
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Generate main-paper ablation, robustness, and polymorph figures from notebook code.')
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'final-figures' / 'standalone-paper' / 'analysis')
    parser.add_argument('--section', nargs='+', choices=['main-ablation', 'appendix-ablation', 'robustness', 'summaries', 'polymorphs', 'all'], default=['all'])
    return parser.parse_args()


def main():
    args = parse_args()
    sections = set(args.section)
    if 'all' in sections:
        sections = {'main-ablation', 'appendix-ablation', 'robustness', 'summaries', 'polymorphs'}
    if 'main-ablation' in sections:
        datasets = load_main_ablation_datasets()
        run_main_ablation_figures(args.output_dir / 'main-ablation', datasets)
        del datasets
        gc.collect()
    if 'appendix-ablation' in sections:
        datasets = load_appendix_ablation_datasets()
        run_appendix_ablation_figures(args.output_dir / 'appendix-ablation', datasets)
        del datasets
        gc.collect()
    if 'robustness' in sections:
        datasets = load_main_ablation_datasets()
        run_main_robustness_figure(args.output_dir / 'robustness', datasets)
        del datasets
        gc.collect()
    if 'summaries' in sections:
        datasets = load_appendix_ablation_datasets()
        run_summary_robustness_figures(args.output_dir / 'summaries', datasets)
        del datasets
        gc.collect()
    if 'polymorphs' in sections:
        run_polymorph_figure(args.output_dir / 'polymorphs', {})
        gc.collect()


if __name__ == '__main__':
    main()
