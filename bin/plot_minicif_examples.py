#!/usr/bin/env python3

import argparse
import gzip
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pymatgen.core import Structure

from bin.test_minicif_realtime import _plot_structure_panel, save_fit_figure
from bin.visualize_minicif import rwp, structure_to_continuous_xrd
from decifer.minicif import minicif_to_structure
from decifer.minicif_v2 import minicif_v2_to_structure


REQUIRED_COLUMNS = {
    "split",
    "sample_index",
    "prompt_mode",
    "reference_minicif",
    "generated_minicif",
    "rwp",
}


def select_best_candidates(
    metrics,
    splits=None,
    prompt_modes=None,
    require_refined=False,
):
    missing = REQUIRED_COLUMNS - set(metrics.columns)
    if missing:
        raise ValueError(f"evaluation metrics are missing columns: {sorted(missing)}")

    candidates = metrics.copy()
    candidates["rwp"] = pd.to_numeric(candidates["rwp"], errors="coerce")
    candidates = candidates.dropna(
        subset=["rwp", "reference_minicif", "generated_minicif"]
    )
    if "structure_ok" in candidates:
        structure_ok = (
            candidates["structure_ok"]
            .astype(str)
            .str.lower()
            .isin({"true", "1"})
        )
        candidates = candidates[structure_ok]
    if splits:
        candidates = candidates[candidates["split"].astype(str).isin(splits)]
    if prompt_modes:
        candidates = candidates[
            candidates["prompt_mode"].astype(str).isin(prompt_modes)
        ]
    if require_refined:
        if "rep" not in candidates:
            raise ValueError(
                "evaluation metrics do not identify candidate repetitions"
            )
        status_column = next(
            (
                column
                for column in ("refined_structure_ok", "refinement_succeeded")
                if column in candidates
            ),
            None,
        )
        if status_column is None:
            raise ValueError(
                "evaluation metrics do not contain refinement results"
            )
        refined_ok = (
            candidates[status_column]
            .astype(str)
            .str.lower()
            .isin({"true", "1"})
        )
        candidates = candidates[refined_ok]
    if candidates.empty:
        raise ValueError("no valid generated structures match the requested filters")

    group_columns = ["split", "sample_index", "prompt_mode"]
    best_indices = candidates.groupby(group_columns, sort=False)["rwp"].idxmin()
    return candidates.loc[best_indices].reset_index(drop=True)


def choose_examples(candidates, count, selection, seed):
    if count <= 0 or count >= len(candidates):
        return candidates.sort_values("rwp").reset_index(drop=True)
    if selection == "best":
        return candidates.nsmallest(count, "rwp").reset_index(drop=True)
    if selection == "worst":
        return candidates.nlargest(count, "rwp").reset_index(drop=True)
    if selection == "crystal-system" and "reference_crystal_system" in candidates:
        rng = np.random.default_rng(seed)
        selected_indices = []
        groups = [
            group.index.to_numpy()
            for _, group in candidates.groupby(
                "reference_crystal_system", sort=True, dropna=True
            )
        ]
        if count < len(groups):
            group_indices = rng.choice(len(groups), size=count, replace=False)
            groups = [groups[index] for index in sorted(group_indices)]
        for indices in groups:
            selected_indices.append(int(rng.choice(indices)))
        remaining = count - len(selected_indices)
        if remaining > 0:
            available = candidates.index.difference(selected_indices).to_numpy()
            selected_indices.extend(
                int(index)
                for index in rng.choice(available, size=remaining, replace=False)
            )
        return candidates.loc[selected_indices].reset_index(drop=True)
    return candidates.sample(n=count, random_state=seed).reset_index(drop=True)


def structure_from_minicif(minicif):
    if str(minicif).lstrip().startswith("<mcif2>"):
        return minicif_v2_to_structure(minicif)
    return minicif_to_structure(minicif)


def load_report(report_dir):
    metrics_path = os.path.join(report_dir, "minicif_generation_metrics.csv")
    summary_path = os.path.join(report_dir, "minicif_summary.json")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(metrics_path)
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(summary_path)
    metrics = pd.read_csv(metrics_path)
    with open(summary_path) as f:
        summary = json.load(f)
    xrd_kwargs = summary.get("xrd_kwargs")
    if not isinstance(xrd_kwargs, dict):
        raise ValueError(f"{summary_path} does not contain xrd_kwargs")
    return metrics, xrd_kwargs


def candidate_identity(row):
    return (
        str(row["split"]),
        int(row["sample_index"]),
        str(row["prompt_mode"]),
        int(row["rep"]),
    )


def load_saved_refinements(report_dir, candidates):
    path = os.path.join(report_dir, "minicif_refinement_results.jsonl.gz")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} is required with --show-refined"
        )
    targets = {
        candidate_identity(row)
        for row in candidates.to_dict(orient="records")
    }
    records = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not record.get("succeeded") or not record.get("refined_cif"):
                continue
            key = candidate_identity(record)
            if key in targets:
                records[key] = record
                if len(records) == len(targets):
                    break
    missing = targets - set(records)
    if missing:
        raise ValueError(
            f"refinement results are missing for {len(missing)} selected candidates"
        )
    return records


def save_refined_example_figure(
    path,
    q_grid,
    reference_iq,
    generated_iq,
    refined_iq,
    reference_structure,
    generated_structure,
    refined_structure,
    sample_name,
    generated_rwp,
    refined_rwp,
    supercell,
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig = plt.figure(figsize=(16, 10), dpi=180, constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=[1.35, 0.55, 2.0])
    ax_fit = fig.add_subplot(grid[0, :])
    ax_residual = fig.add_subplot(grid[1, :], sharex=ax_fit)
    structure_axes = [
        fig.add_subplot(grid[2, index], projection="3d")
        for index in range(3)
    ]

    ax_fit.plot(
        q_grid,
        reference_iq,
        color="#202124",
        linewidth=1.35,
        label="reference",
    )
    ax_fit.plot(
        q_grid,
        generated_iq,
        color="#0072B2",
        linewidth=1.05,
        alpha=0.85,
        label=f"generated, Rwp={generated_rwp:.4f}",
    )
    ax_fit.plot(
        q_grid,
        refined_iq,
        color="#D55E00",
        linewidth=1.05,
        alpha=0.9,
        label=f"refined, Rwp={refined_rwp:.4f}",
    )
    ax_fit.set_ylabel("normalized intensity")
    ax_fit.set_title(f"{sample_name} PXRD fit")
    ax_fit.legend(frameon=False, loc="upper right")
    ax_fit.grid(alpha=0.18)

    ax_residual.axhline(0, color="#444444", linewidth=0.8)
    ax_residual.plot(
        q_grid,
        reference_iq - generated_iq,
        color="#0072B2",
        linewidth=0.85,
        alpha=0.8,
        label="reference - generated",
    )
    ax_residual.plot(
        q_grid,
        reference_iq - refined_iq,
        color="#D55E00",
        linewidth=0.85,
        alpha=0.85,
        label="reference - refined",
    )
    ax_residual.set_xlabel("q")
    ax_residual.set_ylabel("residual")
    ax_residual.legend(frameon=False, loc="upper right")
    ax_residual.grid(alpha=0.18)

    for axis, structure, title in zip(
        structure_axes,
        (reference_structure, generated_structure, refined_structure),
        ("reference structure", "generated structure", "refined structure"),
    ):
        _plot_structure_panel(axis, structure, supercell, title)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_examples(
    candidates,
    xrd_kwargs,
    output_dir,
    wavelength,
    supercell,
    refinement_records=None,
):
    os.makedirs(output_dir, exist_ok=True)
    manifest = []
    for row in candidates.to_dict(orient="records"):
        reference_structure = structure_from_minicif(row["reference_minicif"])
        generated_structure = structure_from_minicif(row["generated_minicif"])
        reference_iq = structure_to_continuous_xrd(
            reference_structure, xrd_kwargs, wavelength
        )
        generated_iq = structure_to_continuous_xrd(
            generated_structure, xrd_kwargs, wavelength
        )
        rendered_rwp = rwp(reference_iq, generated_iq)
        split = str(row["split"])
        prompt_mode = str(row["prompt_mode"])
        sample_index = int(row["sample_index"])
        safe_prompt = re.sub(r"[^A-Za-z0-9_.-]+", "_", prompt_mode)
        path = os.path.join(
            output_dir,
            split,
            f"sample_{sample_index:07d}_{safe_prompt}.png",
        )
        q_grid = (
            float(xrd_kwargs["qmin"])
            + np.arange(len(reference_iq)) * float(xrd_kwargs["qstep"])
        )
        refinement_record = (
            refinement_records.get(candidate_identity(row))
            if refinement_records is not None
            else None
        )
        manifest_row = {
            "split": split,
            "sample_index": sample_index,
            "prompt_mode": prompt_mode,
            "rep": row.get("rep"),
            "evaluation_rwp": row["rwp"],
            "rendered_rwp": rendered_rwp,
            "reference_crystal_system": row.get("reference_crystal_system"),
            "figure_path": os.path.abspath(path),
        }
        if refinement_record is None:
            save_fit_figure(
                path,
                q_grid,
                reference_iq,
                reference_structure,
                [{
                    "rep": row.get("rep", 0),
                    "rwp": rendered_rwp,
                    "generated_iq": generated_iq,
                    "generated_structure": generated_structure,
                }],
                f"{row.get('cif_name', sample_index)} | {prompt_mode}",
                supercell,
            )
        else:
            refined_structure = Structure.from_str(
                refinement_record["refined_cif"],
                fmt="cif",
            )
            refined_iq = structure_to_continuous_xrd(
                refined_structure,
                xrd_kwargs,
                wavelength,
            )
            rendered_refined_rwp = rwp(reference_iq, refined_iq)
            save_refined_example_figure(
                path,
                q_grid,
                reference_iq,
                generated_iq,
                refined_iq,
                reference_structure,
                generated_structure,
                refined_structure,
                f"{row.get('cif_name', sample_index)} | {prompt_mode}",
                rendered_rwp,
                rendered_refined_rwp,
                supercell,
            )
            manifest_row.update({
                "refinement_status": refinement_record.get("status"),
                "evaluation_refined_rwp": row.get("refined_rwp"),
                "rendered_refined_rwp": rendered_refined_rwp,
            })
        manifest.append(manifest_row)
    manifest_path = os.path.join(output_dir, "examples.csv")
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot PXRD and structure examples from a completed minicif evaluation"
    )
    parser.add_argument(
        "--report-dir",
        required=True,
        help="Directory containing minicif_generation_metrics.csv and minicif_summary.json",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory; defaults to REPORT_DIR/evaluation_examples",
    )
    parser.add_argument("--num-examples", type=int, default=7)
    parser.add_argument(
        "--selection",
        choices=["crystal-system", "random", "best", "worst"],
        default="crystal-system",
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--prompt-modes", nargs="+", default=None)
    parser.add_argument("--wavelength", default="CuKa")
    parser.add_argument("--figure-supercell", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--show-refined",
        action="store_true",
        help=(
            "Include the stored refined PXRD and structure; requires "
            "minicif_refinement_results.jsonl.gz"
        ),
    )
    args = parser.parse_args()

    metrics, xrd_kwargs = load_report(args.report_dir)
    candidates = select_best_candidates(
        metrics,
        splits=args.splits,
        prompt_modes=args.prompt_modes,
        require_refined=args.show_refined,
    )
    candidates = choose_examples(
        candidates,
        count=args.num_examples,
        selection=args.selection,
        seed=args.seed,
    )
    output_dir = args.out_dir or os.path.join(
        args.report_dir, "evaluation_examples"
    )
    refinement_records = (
        load_saved_refinements(args.report_dir, candidates)
        if args.show_refined
        else None
    )
    manifest_path = plot_examples(
        candidates,
        xrd_kwargs,
        output_dir,
        args.wavelength,
        args.figure_supercell,
        refinement_records,
    )
    print(f"Wrote {len(candidates)} figures to {os.path.abspath(output_dir)}")
    print(f"Wrote example manifest to {os.path.abspath(manifest_path)}")


if __name__ == "__main__":
    main()
