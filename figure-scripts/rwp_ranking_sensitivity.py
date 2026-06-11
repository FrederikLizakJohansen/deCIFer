#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from decifer.utility import pxrd_from_cif


OUTPUT_DIR = REPO_ROOT / "final-figures" / "revision" / "ranking-sensitivity"


@dataclass(frozen=True)
class MaterialSpec:
    key: str
    title: str
    family: str
    q_min_crop: float
    q_max_crop: float
    protocol_dir: Path
    protocol_prefix: str
    protocol_suffixes: dict[str, str]
    complexity_weight: float = 0.1


MATERIAL_SPECS: dict[str, MaterialSpec] = {
    "ceo2_crystalline": MaterialSpec(
        key="ceo2_crystalline",
        title="Crystalline CeO2",
        family="crystalline",
        q_min_crop=1.5,
        q_max_crop=8.0,
        protocol_dir=REPO_ROOT / "experimental_protocols" / "crystalline_CeO2_protocol_titanrtx",
        protocol_prefix="crystalline_CeO2_BM31_protocol",
        protocol_suffixes={"none": "none", "comp": "Ce4O8", "comp_spg": "Ce4O8_Fm-3m"},
    ),
    "si_crystalline": MaterialSpec(
        key="si_crystalline",
        title="Crystalline Si",
        family="crystalline",
        q_min_crop=1.5,
        q_max_crop=8.0,
        protocol_dir=REPO_ROOT / "experimental_protocols" / "crystalline_Si_protocol_titanrtx",
        protocol_prefix="Si_Mythen_protocol",
        protocol_suffixes={"none": "none", "comp": "Si8", "comp_spg": "Si8_Fd-3m"},
    ),
    "fe2o3_crystalline": MaterialSpec(
        key="fe2o3_crystalline",
        title="Crystalline Fe2O3",
        family="crystalline",
        q_min_crop=0.5,
        q_max_crop=8.0,
        protocol_dir=REPO_ROOT / "experimental_protocols" / "crystalline_Fe2O3_protocol_titanrtx",
        protocol_prefix="AFS012d_a850C_protocol",
        protocol_suffixes={"none": "none", "comp": "Fe12O18", "comp_spg": "Fe12O18_R-3c"},
    ),
    "ceo2_particle": MaterialSpec(
        key="ceo2_particle",
        title="Nanocrystalline CeO2",
        family="particle",
        q_min_crop=1.5,
        q_max_crop=8.0,
        protocol_dir=REPO_ROOT / "experimental_protocols" / "particles_CeO2_protocol_titanrtx",
        protocol_prefix="Hydrolyse_ID10_20min_3-56_boro_0p8_protocol",
        protocol_suffixes={"none": "none", "comp": "Ce4O8", "comp_spg": "Ce4O8_Fm-3m"},
    ),
}


CONDITION_LABELS = {
    "none": "None",
    "comp": "Comp",
    "comp_spg": "Comp + SG",
}


def detailed_settings(family: str) -> tuple[dict[str, Any], ...]:
    if family == "particle":
        return tuple(
            {
                "label": f"f{fwhm:.02f}\ntau={tau}",
                "base_fwhm": fwhm,
                "eta": 0.5,
                "peak_asymmetry": 0.0,
                "particle_size": (None if tau == "none" else float(tau)),
            }
            for fwhm, tau in itertools.product((0.02, 0.05, 0.08), ("none", 5, 10, 20, 40))
        )
    return tuple(
        {
            "label": f"f{fwhm:.02f}\ne{eta:.1f}\na{asym:.1f}",
            "base_fwhm": fwhm,
            "eta": eta,
            "peak_asymmetry": asym,
            "particle_size": None,
        }
        for fwhm, eta, asym in itertools.product((0.03, 0.05, 0.08), (0.3, 0.5, 0.7), (0.0, 0.15))
    )


def paper_settings(family: str) -> tuple[dict[str, Any], ...]:
    if family == "particle":
        return (
            {"label": "f0.02\ntau=none", "base_fwhm": 0.02, "eta": 0.5, "peak_asymmetry": 0.0, "particle_size": None},
            {"label": "f0.02\ntau=10", "base_fwhm": 0.02, "eta": 0.5, "peak_asymmetry": 0.0, "particle_size": 10.0},
            {"label": "f0.02\ntau=20", "base_fwhm": 0.02, "eta": 0.5, "peak_asymmetry": 0.0, "particle_size": 20.0},
            {"label": "f0.05\ntau=none", "base_fwhm": 0.05, "eta": 0.5, "peak_asymmetry": 0.0, "particle_size": None},
            {"label": "f0.05\ntau=10", "base_fwhm": 0.05, "eta": 0.5, "peak_asymmetry": 0.0, "particle_size": 10.0},
            {"label": "f0.05\ntau=20", "base_fwhm": 0.05, "eta": 0.5, "peak_asymmetry": 0.0, "particle_size": 20.0},
        )
    return (
        {"label": "f0.03\ne0.5\na0.0", "base_fwhm": 0.03, "eta": 0.5, "peak_asymmetry": 0.0, "particle_size": None},
        {"label": "f0.03\ne0.5\na0.15", "base_fwhm": 0.03, "eta": 0.5, "peak_asymmetry": 0.15, "particle_size": None},
        {"label": "f0.05\ne0.5\na0.0", "base_fwhm": 0.05, "eta": 0.5, "peak_asymmetry": 0.0, "particle_size": None},
        {"label": "f0.05\ne0.5\na0.15", "base_fwhm": 0.05, "eta": 0.5, "peak_asymmetry": 0.15, "particle_size": None},
        {"label": "f0.08\ne0.5\na0.0", "base_fwhm": 0.08, "eta": 0.5, "peak_asymmetry": 0.0, "particle_size": None},
        {"label": "f0.08\ne0.5\na0.15", "base_fwhm": 0.08, "eta": 0.5, "peak_asymmetry": 0.15, "particle_size": None},
    )


def paper_fe_tau_settings(family: str) -> tuple[dict[str, Any], ...]:
    if family == "particle":
        return (
            {"label": "f0.02\ntau=none", "base_fwhm": 0.02, "eta": 0.5, "peak_asymmetry": 0.0, "particle_size": None},
            {"label": "f0.02\ntau=10", "base_fwhm": 0.02, "eta": 0.5, "peak_asymmetry": 0.0, "particle_size": 10.0},
            {"label": "f0.02\ntau=20", "base_fwhm": 0.02, "eta": 0.5, "peak_asymmetry": 0.0, "particle_size": 20.0},
            {"label": "f0.05\ntau=none", "base_fwhm": 0.05, "eta": 0.5, "peak_asymmetry": 0.0, "particle_size": None},
            {"label": "f0.05\ntau=10", "base_fwhm": 0.05, "eta": 0.5, "peak_asymmetry": 0.0, "particle_size": 10.0},
            {"label": "f0.05\ntau=20", "base_fwhm": 0.05, "eta": 0.5, "peak_asymmetry": 0.0, "particle_size": 20.0},
            {"label": "f0.08\ntau=none", "base_fwhm": 0.08, "eta": 0.5, "peak_asymmetry": 0.0, "particle_size": None},
            {"label": "f0.08\ntau=10", "base_fwhm": 0.08, "eta": 0.5, "peak_asymmetry": 0.0, "particle_size": 10.0},
            {"label": "f0.08\ntau=20", "base_fwhm": 0.08, "eta": 0.5, "peak_asymmetry": 0.0, "particle_size": 20.0},
        )
    return tuple(
        {
            "label": f"f{fwhm:.02f}\ne{eta:.1f}",
            "base_fwhm": fwhm,
            "eta": eta,
            "peak_asymmetry": 0.0,
            "particle_size": None,
        }
        for fwhm, eta in itertools.product((0.03, 0.05, 0.08), (0.3, 0.5, 0.7))
    )


def protocol_path(material: MaterialSpec, condition: str) -> Path:
    suffix = material.protocol_suffixes[condition]
    return material.protocol_dir / f"{material.protocol_prefix}_{suffix}.pkl"


def case_key(material: str, condition: str) -> str:
    return f"{material}__{condition}"


def split_case_key(case_name: str) -> tuple[str, str]:
    if "__" in case_name:
        material_key, condition_key = case_name.split("__", 1)
        return material_key, condition_key
    if case_name.endswith("_none"):
        material_key = case_name[: -len("_none")]
        return material_key, "none"
    return case_name, "none"


def enrich_loaded_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parsed = out["case_key"].map(split_case_key)
    material_keys = [p[0] for p in parsed]
    condition_keys = [p[1] for p in parsed]
    out["case_key"] = [case_key(m, c) for m, c in zip(material_keys, condition_keys, strict=False)]
    if "material_key" not in out.columns or "condition_key" not in out.columns:
        out["material_key"] = material_keys
        out["condition_key"] = condition_keys
    if "condition_label" not in out.columns:
        out["condition_label"] = out["condition_key"].map(CONDITION_LABELS).fillna(out["condition_key"])
    if "material_title" not in out.columns:
        out["material_title"] = out["material_key"].map(lambda key: MATERIAL_SPECS[key].title if key in MATERIAL_SPECS else key)
    if "protocol_path" not in out.columns:
        out["protocol_path"] = ""
    return out


def build_case_setting_labels(case_scores: pd.DataFrame) -> pd.DataFrame:
    material_key = str(case_scores["material_key"].iloc[0])
    family = MATERIAL_SPECS[material_key].family if material_key in MATERIAL_SPECS else "crystalline"
    settings = (
        case_scores[["setting_index", "base_fwhm", "eta", "peak_asymmetry", "particle_size"]]
        .drop_duplicates()
        .sort_values("setting_index")
        .copy()
    )

    vary_f = settings["base_fwhm"].nunique() > 1
    vary_e = settings["eta"].nunique() > 1
    vary_a = settings["peak_asymmetry"].nunique() > 1
    vary_tau = settings["particle_size"].fillna(-1).nunique() > 1

    labels = []
    for _, row in settings.iterrows():
        parts = []
        if vary_f or family in {"crystalline", "particle"}:
            parts.append(f"f{float(row['base_fwhm']):.02f}")
        if family == "particle":
            if vary_tau:
                if pd.isna(row["particle_size"]):
                    parts.append("tau=none")
                else:
                    parts.append(f"tau={int(float(row['particle_size']))}")
        else:
            if vary_e:
                parts.append(f"e{float(row['eta']):.1f}")
            if vary_a:
                parts.append(f"a{float(row['peak_asymmetry']):.1f}")
        labels.append("\n".join(parts))

    settings["display_setting_label"] = labels
    return settings[["setting_index", "display_setting_label"]]


def load_protocol(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def cropped_pattern(protocol: dict[str, Any], q_min_crop: float, q_max_crop: float) -> tuple[np.ndarray, np.ndarray]:
    exp_q = np.asarray(protocol["exp_q"])
    exp_i = np.asarray(protocol["exp_i"])
    mask = (exp_q >= q_min_crop) & (exp_q <= q_max_crop)
    return exp_q[mask], exp_i[mask]


def score_candidate(
    cif_str: str,
    q_target: np.ndarray,
    i_target: np.ndarray,
    *,
    complexity_weight: float,
    base_fwhm: float,
    eta: float,
    peak_asymmetry: float,
    particle_size: float | None,
) -> dict[str, float]:
    pxrd = pxrd_from_cif(
        cif_str,
        qmin=float(q_target.min()),
        qmax=float(q_target.max()) + 1e-6,
        qstep=float(np.median(np.diff(q_target))),
        base_fwhm=base_fwhm,
        eta=eta,
        particle_size=particle_size,
        peak_asymmetry=peak_asymmetry,
    )
    if pxrd is None:
        return {"rwp": np.inf, "n_peaks": np.inf, "ranking_score": np.inf}

    i_pred = np.interp(q_target, pxrd["q"], pxrd["iq"])
    rwp = float(np.sqrt(np.sum(np.square(i_target - i_pred)) / np.sum(np.square(i_target))))
    n_peaks = int(len(pxrd["q_disc"][0]))
    ranking_score = float(rwp + complexity_weight * n_peaks)
    return {"rwp": rwp, "n_peaks": n_peaks, "ranking_score": ranking_score}


def evaluate_case(
    material: MaterialSpec,
    condition: str,
    settings: tuple[dict[str, Any], ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = protocol_path(material, condition)
    protocol = load_protocol(path)
    q_target, i_target = cropped_pattern(protocol, material.q_min_crop, material.q_max_crop)
    candidates = protocol["results"]["gens"]

    rows: list[dict[str, Any]] = []
    ranking_vectors: dict[str, pd.Series] = {}

    for setting_index, setting in enumerate(
        tqdm(settings, desc=f"{material.key}:{condition}", leave=False, dynamic_ncols=True)
    ):
        frame_rows: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidates):
            scores = score_candidate(
                candidate["cif_str"],
                q_target,
                i_target,
                complexity_weight=material.complexity_weight,
                base_fwhm=setting["base_fwhm"],
                eta=setting["eta"],
                peak_asymmetry=setting["peak_asymmetry"],
                particle_size=setting["particle_size"],
            )
            frame_rows.append(
                {
                    "case_key": case_key(material.key, condition),
                    "material_key": material.key,
                    "material_title": material.title,
                    "condition_key": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "protocol_path": str(path.relative_to(REPO_ROOT)),
                    "setting_index": setting_index,
                    "setting_label": setting["label"],
                    "candidate_index": candidate_index,
                    "base_fwhm": setting["base_fwhm"],
                    "eta": setting["eta"],
                    "peak_asymmetry": setting["peak_asymmetry"],
                    "particle_size": setting["particle_size"],
                    **scores,
                }
            )

        frame = pd.DataFrame(frame_rows).sort_values(
            ["ranking_score", "candidate_index"], ascending=[True, True]
        ).reset_index(drop=True)
        frame["rank"] = np.arange(1, len(frame) + 1)
        rows.extend(frame.to_dict("records"))
        ranking_vectors[setting["label"]] = frame.set_index("candidate_index")["rank"].sort_index()

    scores_df = pd.DataFrame(rows)
    baseline_label = settings[0]["label"]
    baseline_ranks = ranking_vectors[baseline_label]
    baseline_top5 = set(baseline_ranks.nsmallest(5).index.tolist())
    baseline_top10 = set(baseline_ranks.nsmallest(10).index.tolist())
    baseline_top1 = int(baseline_ranks.idxmin())

    summary_rows: list[dict[str, Any]] = []
    for setting in settings:
        label = setting["label"]
        ranks = ranking_vectors[label]
        top1 = int(ranks.idxmin())
        top5 = set(ranks.nsmallest(5).index.tolist())
        top10 = set(ranks.nsmallest(10).index.tolist())
        summary_rows.append(
            {
                "case_key": case_key(material.key, condition),
                "material_key": material.key,
                "material_title": material.title,
                "condition_key": condition,
                "condition_label": CONDITION_LABELS[condition],
                "protocol_path": str(path.relative_to(REPO_ROOT)),
                "setting_label": label,
                "top1_candidate_index": top1,
                "top1_matches_baseline": top1 == baseline_top1,
                "top5_overlap_with_baseline": len(top5 & baseline_top5),
                "top10_overlap_with_baseline": len(top10 & baseline_top10),
                "spearman_vs_baseline": float(spearmanr(baseline_ranks.values, ranks.values).statistic),
            }
        )

    return scores_df, pd.DataFrame(summary_rows)


def aggregate_case_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    return (
        summary_df.groupby(
            ["case_key", "material_key", "material_title", "condition_key", "condition_label", "protocol_path"],
            as_index=False,
        )
        .agg(
            top1_stable_fraction=("top1_matches_baseline", "mean"),
            min_top5_overlap=("top5_overlap_with_baseline", "min"),
            mean_top5_overlap=("top5_overlap_with_baseline", "mean"),
            min_top10_overlap=("top10_overlap_with_baseline", "min"),
            mean_top10_overlap=("top10_overlap_with_baseline", "mean"),
            min_spearman=("spearman_vs_baseline", "min"),
            mean_spearman=("spearman_vs_baseline", "mean"),
        )
    )


def save_per_case_csvs(scores_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> None:
    for case_key_value, sub in scores_df.groupby("case_key"):
        sub.to_csv(output_dir / f"{case_key_value}_scores.csv", index=False)
    for case_key_value, sub in summary_df.groupby("case_key"):
        sub.to_csv(output_dir / f"{case_key_value}_summary.csv", index=False)


def per_case_scores_path(output_dir: Path, case_name: str) -> Path:
    return output_dir / f"{case_name}_scores.csv"


def per_case_summary_path(output_dir: Path, case_name: str) -> Path:
    return output_dir / f"{case_name}_summary.csv"


def case_checkpoint_exists(output_dir: Path, case_name: str) -> bool:
    return per_case_scores_path(output_dir, case_name).exists() and per_case_summary_path(output_dir, case_name).exists()


def load_case_checkpoint(output_dir: Path, case_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores_df = enrich_loaded_frame(pd.read_csv(per_case_scores_path(output_dir, case_name)))
    summary_df = enrich_loaded_frame(pd.read_csv(per_case_summary_path(output_dir, case_name)))
    return scores_df, summary_df


def write_case_checkpoint(output_dir: Path, scores_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    case_name = str(scores_df["case_key"].iloc[0])
    scores_df.to_csv(per_case_scores_path(output_dir, case_name), index=False)
    summary_df.to_csv(per_case_summary_path(output_dir, case_name), index=False)


def plot_mockup(
    all_scores: pd.DataFrame,
    all_summary: pd.DataFrame,
    case_order: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        nrows=len(case_order),
        ncols=2,
        figsize=(15, 4.4 * len(case_order)),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2.4, 1.0]},
    )
    if len(case_order) == 1:
        axes = np.array([axes])

    for row_index, case_name in enumerate(case_order):
        case_scores = all_scores[all_scores["case_key"] == case_name].copy()
        case_summary = all_summary[all_summary["case_key"] == case_name].copy()
        setting_order = build_case_setting_labels(case_scores)
        baseline_setting_index = int(case_scores["setting_index"].min())
        title = f'{case_scores["material_title"].iloc[0]} ({case_scores["condition_label"].iloc[0]})'
        baseline_order = (
            case_scores[case_scores["setting_index"] == baseline_setting_index]
            .sort_values("rank")["candidate_index"]
            .head(10)
            .tolist()
        )
        heatmap = (
            case_scores[case_scores["candidate_index"].isin(baseline_order)]
            .pivot(index="candidate_index", columns="setting_index", values="rank")
            .loc[baseline_order, setting_order["setting_index"].tolist()]
        )

        heat_ax = axes[row_index, 0]
        im = heat_ax.imshow(heatmap.values, aspect="auto", cmap="viridis_r", vmin=1, vmax=max(10, len(baseline_order)))
        heat_ax.set_title(f"{title}: rank of baseline top-10 candidates", fontsize=11)
        heat_ax.set_xticks(np.arange(len(heatmap.columns)))
        heat_ax.set_xticklabels(setting_order["display_setting_label"].tolist(), rotation=90, fontsize=7)
        heat_ax.set_yticks(np.arange(len(heatmap.index)))
        heat_ax.set_yticklabels([f"cand {idx}" for idx in heatmap.index], fontsize=8)
        heat_ax.set_xlabel("Peak-shape setting")
        heat_ax.set_ylabel("Baseline-ranked candidates")
        cbar = fig.colorbar(im, ax=heat_ax, fraction=0.03, pad=0.02)
        cbar.set_label("Rank")

        summary_ax = axes[row_index, 1]
        x = np.arange(len(case_summary))
        summary_ax.plot(x, case_summary["spearman_vs_baseline"], marker="o", label="Spearman vs baseline")
        summary_ax.plot(x, case_summary["top5_overlap_with_baseline"] / 5.0, marker="s", label="Top-5 overlap / 5")
        summary_ax.scatter(x, case_summary["top1_matches_baseline"].astype(int), marker="x", s=45, label="Top-1 unchanged")
        summary_ax.set_ylim(-0.05, 1.05)
        summary_ax.set_title(f"{title}: stability summary", fontsize=11)
        summary_ax.set_xticks(x)
        summary_ax.set_xticklabels(case_summary["setting_label"], rotation=90, fontsize=7)
        summary_ax.set_xlabel("Peak-shape setting")
        summary_ax.legend(frameon=False, fontsize=7, loc="lower left")

    fig.suptitle("Mockup: ranking sensitivity to PXRD peak-shape assumptions", fontsize=15)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_paper_heatmaps(
    all_scores: pd.DataFrame,
    material_order: list[str],
    condition_order: list[str],
    output_path: Path,
) -> None:
    axis_label_fontsize = 12
    tick_label_fontsize = 10
    title_fontsize = 12
    suptitle_fontsize = 14
    cmap = plt.get_cmap("YlGnBu")

    fig, axes = plt.subplots(
        nrows=len(material_order),
        ncols=len(condition_order),
        figsize=(11.6, 8.8),
        constrained_layout=True,
    )
    if len(material_order) == 1 and len(condition_order) == 1:
        axes = np.array([[axes]])
    elif len(material_order) == 1:
        axes = axes[np.newaxis, :]
    elif len(condition_order) == 1:
        axes = axes[:, np.newaxis]

    last_im = None
    for row_idx, material_key in enumerate(material_order):
        for col_idx, condition_key in enumerate(condition_order):
            ax = axes[row_idx, col_idx]
            case_name = case_key(material_key, condition_key)
            case_scores = all_scores[all_scores["case_key"] == case_name].copy()
            if case_scores.empty:
                ax.axis("off")
                continue
            setting_order = build_case_setting_labels(case_scores)
            baseline_setting_index = int(case_scores["setting_index"].min())
            baseline_order = (
                case_scores[case_scores["setting_index"] == baseline_setting_index]
                .sort_values("rank")["candidate_index"]
                .head(5)
                .tolist()
            )
            heatmap = (
                case_scores[case_scores["candidate_index"].isin(baseline_order)]
                .pivot(index="candidate_index", columns="setting_index", values="rank")
                .loc[baseline_order, setting_order["setting_index"].tolist()]
            )
            vmax = max(5, len(baseline_order))
            last_im = ax.imshow(heatmap.values, aspect="auto", cmap=cmap, vmin=1, vmax=vmax)
            if row_idx == 0:
                ax.set_title(CONDITION_LABELS[condition_key], fontsize=title_fontsize, pad=9)
            if col_idx == 0:
                ax.set_ylabel(MATERIAL_SPECS[material_key].title + "\nBaseline top-5", fontsize=axis_label_fontsize)
            ax.set_xticks(np.arange(len(heatmap.columns)))
            ax.set_xticklabels(setting_order["display_setting_label"].tolist(), rotation=90, fontsize=tick_label_fontsize)
            ax.set_yticks(np.arange(len(heatmap.index)))
            ax.set_yticklabels([str(idx) for idx in heatmap.index], fontsize=tick_label_fontsize)
            ax.tick_params(axis="both", labelsize=tick_label_fontsize)

            midpoint = 0.5 * (1 + vmax)
            for row_i in range(heatmap.shape[0]):
                for col_i in range(heatmap.shape[1]):
                    value = heatmap.iat[row_i, col_i]
                    text_color = "white" if value >= midpoint else "black"
                    ax.text(
                        col_i,
                        row_i,
                        f"{int(value)}",
                        ha="center",
                        va="center",
                        fontsize=9.5,
                        color=text_color,
                    )

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
        cbar.set_label("Rank", fontsize=axis_label_fontsize)
        cbar.ax.tick_params(labelsize=tick_label_fontsize)
    fig.suptitle("Ranking sensitivity across conditioning choices", fontsize=suptitle_fontsize)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_paper_summary(case_metrics: pd.DataFrame, material_order: list[str], condition_order: list[str], output_path: Path) -> None:
    axis_label_fontsize = 12
    tick_label_fontsize = 10
    title_fontsize = 12
    suptitle_fontsize = 14
    cmap = plt.get_cmap("YlGnBu")

    metric_names = [
        ("top1_stable_fraction", "Top-1 stable"),
        ("min_top5_overlap", "Min top-5 overlap"),
        ("min_spearman", "Min Spearman"),
    ]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(10.8, 6.0), constrained_layout=True)
    if len(metric_names) == 1:
        axes = [axes]

    matrix_index = [case_key(material, condition) for material in material_order for condition in condition_order]
    labels = [f"{MATERIAL_SPECS[material].title}\n{CONDITION_LABELS[condition]}" for material in material_order for condition in condition_order]
    indexed = case_metrics.set_index("case_key")

    for ax, (metric_key, metric_label) in zip(axes, metric_names, strict=False):
        values = np.array([[indexed.loc[row_key, metric_key] if row_key in indexed.index else np.nan] for row_key in matrix_index])
        im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(metric_label, fontsize=title_fontsize, pad=9)
        ax.set_xticks([0])
        ax.set_xticklabels([metric_label], rotation=90, fontsize=tick_label_fontsize)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels if ax is axes[0] else [""] * len(labels), fontsize=tick_label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)
        for y, row_key in enumerate(matrix_index):
            if row_key in indexed.index:
                val = indexed.loc[row_key, metric_key]
                display_val = f"{val:.2f}" if "overlap" not in metric_key else f"{val:.1f}"
                ax.text(0, y, display_val, ha="center", va="center", color="white", fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.03)
        cbar.ax.tick_params(labelsize=tick_label_fontsize)

    fig.suptitle("Supplementary summary of ranking stability", fontsize=suptitle_fontsize)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["score", "plot", "all"],
        default="all",
        help="'score' computes CSVs only, 'plot' reuses existing CSVs only, 'all' does both.",
    )
    parser.add_argument(
        "--materials",
        nargs="+",
        default=["ceo2_crystalline", "si_crystalline", "fe2o3_crystalline", "ceo2_particle"],
        choices=sorted(MATERIAL_SPECS.keys()),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["none", "comp", "comp_spg"],
        choices=["none", "comp", "comp_spg"],
    )
    parser.add_argument(
        "--profile",
        choices=["detailed", "paper", "paper_fe_tau"],
        default="detailed",
        help="Parameter sweep profile. 'paper' uses a smaller setting grid; 'paper_fe_tau' removes asymmetry and varies f/e or f/tau.",
    )
    parser.add_argument(
        "--plot-style",
        nargs="+",
        default=["mockup", "paper"],
        choices=["mockup", "paper", "none"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing precomputed ranking_sensitivity_*.csv files for plot mode. Defaults to --output-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    input_dir: Path = args.input_dir or output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_materials = [MATERIAL_SPECS[key] for key in args.materials]
    case_order: list[str] = [case_key(material.key, condition) for material in selected_materials for condition in args.conditions]

    if args.mode in {"score", "all"}:
        settings_lookup = {
            spec.key: (
                paper_settings(spec.family)
                if args.profile == "paper"
                else paper_fe_tau_settings(spec.family)
                if args.profile == "paper_fe_tau"
                else detailed_settings(spec.family)
            )
            for spec in selected_materials
        }

        score_frames = []
        summary_frames = []
        total_cases = len(selected_materials) * len(args.conditions)
        with tqdm(total=total_cases, desc="Cases", dynamic_ncols=True) as pbar_cases:
            for material in selected_materials:
                for condition in args.conditions:
                    case_name = case_key(material.key, condition)
                    if case_checkpoint_exists(output_dir, case_name):
                        scores_df, summary_df = load_case_checkpoint(output_dir, case_name)
                    else:
                        scores_df, summary_df = evaluate_case(material, condition, settings_lookup[material.key])
                        write_case_checkpoint(output_dir, scores_df, summary_df)
                    score_frames.append(scores_df)
                    summary_frames.append(summary_df)
                    pbar_cases.update(1)

        all_scores = pd.concat(score_frames, ignore_index=True)
        all_summary = pd.concat(summary_frames, ignore_index=True)
        case_metrics = aggregate_case_metrics(all_summary)

        all_scores.to_csv(output_dir / "ranking_sensitivity_scores.csv", index=False)
        all_summary.to_csv(output_dir / "ranking_sensitivity_summary.csv", index=False)
        case_metrics.to_csv(output_dir / "ranking_sensitivity_case_metrics.csv", index=False)
    else:
        all_scores = enrich_loaded_frame(pd.read_csv(input_dir / "ranking_sensitivity_scores.csv"))
        all_summary = enrich_loaded_frame(pd.read_csv(input_dir / "ranking_sensitivity_summary.csv"))
        case_metrics_path = input_dir / "ranking_sensitivity_case_metrics.csv"
        if case_metrics_path.exists():
            case_metrics = enrich_loaded_frame(pd.read_csv(case_metrics_path))
        else:
            case_metrics = aggregate_case_metrics(all_summary)
        all_scores = all_scores[
            all_scores["material_key"].isin(args.materials) & all_scores["condition_key"].isin(args.conditions)
        ].copy()
        all_summary = all_summary[
            all_summary["material_key"].isin(args.materials) & all_summary["condition_key"].isin(args.conditions)
        ].copy()
        case_metrics = case_metrics[
            case_metrics["material_key"].isin(args.materials) & case_metrics["condition_key"].isin(args.conditions)
        ].copy()

    plot_styles = [] if args.plot_style == ["none"] else args.plot_style
    if "mockup" in plot_styles:
        plot_mockup(all_scores, all_summary, case_order, output_dir / "ranking_sensitivity_mockup.png")
    if "paper" in plot_styles:
        plot_paper_heatmaps(
            all_scores,
            material_order=args.materials,
            condition_order=args.conditions,
            output_path=output_dir / "ranking_sensitivity_paper_heatmaps.png",
        )
        plot_paper_summary(
            case_metrics,
            material_order=args.materials,
            condition_order=args.conditions,
            output_path=output_dir / "ranking_sensitivity_paper_summary.png",
        )


if __name__ == "__main__":
    main()
