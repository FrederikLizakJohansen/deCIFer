#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from ase.data import atomic_numbers, colors
from ase.visualize.plot import plot_atoms
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from decifer.utility import extract_formula_nonreduced, extract_space_group_symbol


REPO_ROOT = Path(__file__).resolve().parent
PKL_DIR = REPO_ROOT / "pkl-files"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "final-figures" / "revision"


color_data = "black"
color_prediction_stem = "#d33f49"
color_prediction_label = color_prediction_stem
color_particles_fit = "#3f88c5"
atom_edge_color = "black"
legend_text_color = "white"
stem_linestyle = "-"
stem_linewidth = 1.0
marker_size = 3.0
pred_label_size = 12
atoms_radius = 48
atoms_label_size = 8
tx = 0.98
ty = 0.88
scherrer_k = 0.9
scherrer_tau_angstrom = 10.0


def format_formula_latex(formula: str) -> str:
    formatted = re.sub(r"(\D)(\d+)", r"\1_{\2}", formula)
    return rf"$\mathrm{{{formatted}}}$"


def convert_space_group_to_latex(space_group: str) -> str:
    latex_str = re.sub(r"-(\d+)", r"\\bar{\1}", space_group)
    return f"${latex_str}$"


def plot_unit_cell_with_boundaries(
    structure: Structure,
    *,
    ax: plt.Axes,
    tol: float = 1e-5,
    radii: float = 0.5,
    rotation: str = "45x, -15y, 90z",
    offset: tuple[float, float, float] = (0, 0, 0),
) -> None:
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

    all_species: list[str] = []
    all_coords: list[np.ndarray] = []

    for tv in translation_vectors:
        tv_cart = structure.lattice.get_cartesian_coords(tv)
        for site in structure:
            if tv == [0, 0, 0]:
                all_species.append(site.species_string)
                all_coords.append(site.coords)
            elif all(site.frac_coords[i] < tol for i, shift in enumerate(tv) if shift == 1):
                all_species.append(site.species_string)
                all_coords.append(site.coords + tv_cart)

    discrete_structure = Structure(
        lattice=structure.lattice.matrix,
        species=all_species,
        coords=np.array(all_coords),
        coords_are_cartesian=True,
    )

    ase_atoms = AseAtomsAdaptor.get_atoms(discrete_structure)
    ase_atoms.set_pbc([False, False, False])
    plot_atoms(ase_atoms, ax, radii=radii, show_unit_cell=True, rotation=rotation, offset=offset)


def adjust_unit_cell_view(ax: plt.Axes, *, zoom: float = 1.0, dx: float = 0.0, dy: float = 0.0) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_center, y_center = (x0 + x1) / 2, (y0 + y1) / 2
    x_half = (x1 - x0) / 2 * zoom
    y_half = (y1 - y0) / 2 * zoom
    ax.set_xlim(x_center - x_half + dx, x_center + x_half + dx)
    ax.set_ylim(y_center - y_half + dy, y_center + y_half + dy)


def fixed_size_circle(x: float, y: float, radius_points: float, ax: plt.Axes, **kwargs: Any) -> Circle:
    inv = ax.transData.inverted()
    dp = inv.transform((radius_points, 0)) - inv.transform((0, 0))
    data_radius = dp[0]
    return Circle((x, y), data_radius, **kwargs)


def get_atom_color(species: str) -> Any:
    try:
        return colors.jmol_colors[atomic_numbers[species]]
    except (KeyError, TypeError):
        return "black"


def scherrer_broaden_in_q(
    q_grid: np.ndarray,
    q_disc: np.ndarray,
    iq_disc: np.ndarray,
    *,
    tau_angstrom: float = scherrer_tau_angstrom,
    shape_factor: float = scherrer_k,
) -> np.ndarray:
    # Match the manuscript's nanoparticle approximation convention in q-space.
    fwhm_q = shape_factor / tau_angstrom
    gamma = fwhm_q / 2
    delta_q = q_grid[:, None] - q_disc[None, :]
    lorentz = 1.0 / (1.0 + (delta_q / gamma) ** 2)
    iq_cont = (lorentz * iq_disc[None, :]).sum(axis=1)
    return iq_cont / max(iq_cont.max(), 1e-12)


def disable_axis_clipping(ax: plt.Axes) -> None:
    for artist in list(ax.lines) + list(ax.patches) + list(ax.collections) + list(ax.artists):
        try:
            artist.set_clip_on(False)
        except Exception:
            pass


def save_figure_outputs(fig: plt.Figure, output_filename: Path) -> None:
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_filename, bbox_inches="tight")
    fig.savefig(output_filename.with_suffix(".png"), bbox_inches="tight")


def plot_pxrd_structures(
    rows: list[dict[str, Any]],
    *,
    output_filename: Path,
    figsize: float = 1.2,
    dpi: int = 300,
    title_fontsize: int = 10,
    title_struc_fontsize: int = 9,
    hspace: float = 0.4,
    wspace: float = -0.1,
    bbox_legend: tuple[float, float] = (0.5, 1.6),
    show_atom_legend: bool = True,
) -> None:
    nrows = len(rows)
    fig = plt.figure(figsize=(5.5 * figsize, 1.2 * nrows * figsize), dpi=dpi)
    gs = gridspec.GridSpec(
        nrows,
        2,
        width_ratios=[3, 1],
        height_ratios=[1] * nrows,
        hspace=hspace,
        wspace=wspace,
    )

    first_pxrd_ax = None

    for i, row in enumerate(rows):
        ar = row.get("atoms_radius", atoms_radius)
        als = row.get("atoms_label_size", atoms_label_size)

        ax_pxrd = fig.add_subplot(gs[i, 0], sharex=first_pxrd_ax if first_pxrd_ax else None)
        ax_pxrd.set_title(row["title"], fontsize=title_fontsize)

        stem = ax_pxrd.stem(
            row["q_disc"],
            row["iq_disc"],
            linefmt="-",
            markerfmt="o",
            basefmt=" ",
            label=row.get("label", "best prediction"),
        )
        plt.setp(
            stem.stemlines,
            color=color_prediction_stem,
            linestyle=stem_linestyle,
            linewidth=stem_linewidth,
            zorder=2,
        )
        plt.setp(
            stem.markerline,
            markersize=marker_size,
            markerfacecolor="white",
            markeredgecolor=color_prediction_stem,
            markeredgewidth=1.0,
            zorder=2,
        )
        plt.setp(stem.baseline, zorder=2)

        ax_pxrd.plot(row["q"], row["iq"], lw=1, c=color_data, alpha=0.9, zorder=3, label="data")

        ax_pxrd.set(xlim=(1.5, 7.5), ylim=(0, None), yticklabels=[], yticks=[], ylabel=r"$I(Q)_{\;[a.u.]}$")
        ax_pxrd.grid(alpha=0.2, axis="x")
        if i < nrows - 1:
            ax_pxrd.tick_params(axis="x", labelbottom=False)
        else:
            ax_pxrd.set_xlabel(r"$Q_{\;[\AA^{-1}]}$", fontsize=11)
        ax_pxrd.tick_params(axis="both", labelsize=10)
        ax_pxrd.yaxis.label.set_size(11)

        if first_pxrd_ax is None:
            first_pxrd_ax = ax_pxrd

        ax_struc = fig.add_subplot(gs[i, 1])
        ax_struc.axis("off")
        if "title_struc" in row:
            ax_struc.set_title(row["title_struc"], fontsize=title_struc_fontsize)

        plot_unit_cell_with_boundaries(row["structure"], ax=ax_struc, radii=0.5)
        ylim = ax_struc.get_ylim()
        ax_struc.set_ylim((ylim[0] - 4, ylim[1]))

        view = row.get("view", {})
        adjust_unit_cell_view(
            ax_struc,
            zoom=view.get("zoom", 1.0),
            dx=view.get("dx", 0.0),
            dy=view.get("dy", 0.0),
        )

        if show_atom_legend:
            unique_species = sorted({site.species_string for site in row["structure"]})
            x_positions = np.linspace(*ax_struc.get_xlim(), len(unique_species) + 2)[1:-1]
            offset_y = view.get("atom_offset_y", 0.0)
            legend_y = ax_struc.get_ylim()[0] + 2.0 + offset_y

            for x, species in zip(x_positions, unique_species):
                atom_color = get_atom_color(species)
                circ = fixed_size_circle(
                    x,
                    legend_y,
                    radius_points=ar,
                    ax=ax_struc,
                    edgecolor=atom_edge_color,
                    facecolor=atom_color,
                    lw=1,
                )
                ax_struc.add_patch(circ)
                ax_struc.text(x, legend_y, species, color=legend_text_color, ha="center", va="center", fontsize=als)

        ax_pxrd.text(
            tx,
            ty,
            f"{row['formula_latex']} {row['spacegroup_latex']}",
            transform=ax_pxrd.transAxes,
            ha="right",
            va="center",
            fontsize=pred_label_size,
            color=color_prediction_label,
        )

        if i == 0:
            ax_pxrd.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=bbox_legend, fontsize=10)

    save_figure_outputs(fig, output_filename)
    plt.close(fig)


def build_rows_from_pickle(
    pkl_filename: str,
    key_config_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    with (PKL_DIR / pkl_filename).open("rb") as handle:
        loaded = pickle.load(handle)

    rows: list[dict[str, Any]] = []
    for key, cfg in key_config_map.items():
        data = loaded[key]
        cif_str = data["result"]["cif_str"]
        formula_rd = "".join(extract_formula_nonreduced(cif_str).split())

        row: dict[str, Any] = {
            "q": data["exp_q"],
            "iq": data["exp_i"],
            "q_disc": np.array(data["pxrd"]["q_disc"][0]),
            "iq_disc": np.array(data["pxrd"]["iq_disc"][0]) / 100,
            "structure": data["result"]["struct"],
            "formula_latex": format_formula_latex(formula_rd),
            "spacegroup_latex": convert_space_group_to_latex(extract_space_group_symbol(cif_str)),
            "title": cfg["title"],
        }

        if "atoms_radius" in cfg:
            row["atoms_radius"] = cfg["atoms_radius"]
        if "atoms_label_size" in cfg:
            row["atoms_label_size"] = cfg["atoms_label_size"]

        view = {}
        for view_key in ("zoom", "dx", "dy", "atom_offset_y"):
            if view_key in cfg:
                view[view_key] = cfg[view_key]
        if view:
            row["view"] = view

        rows.append(row)

    return rows


APPENDIX_FIGURE_SPECS: dict[str, dict[str, Any]] = {
    "s34_ceo2_crystalline": {
        "pkl": "crystalline_CeO2.pkl",
        "output": "revision_main_ceo2_crystalline.pdf",
        "config": {
            "crystalline_CeO2_BM31_protocol_Ce1O2": {
                "title": r"Input to model: [composition: Ce$_1$O$_2$, spacegroup: None]",
                "atoms_radius": 60,
                "atoms_label_size": 6,
                "zoom": 1.2,
            },
            "crystalline_CeO2_BM31_protocol_Ce2O4": {
                "title": r"Input to model: [composition: Ce$_2$O$_4$, spacegroup: None]",
            },
            "crystalline_CeO2_BM31_protocol_Ce4O8": {
                "title": r"Input to model: [composition: Ce$_4$O$_8$, spacegroup: None]",
            },
            "crystalline_CeO2_BM31_protocol_Ce4O8_Fm-3m": {
                "title": r"Input to model: [composition: Ce$_4$O$_8$, spacegroup: $Fm\bar{3}m$]",
            },
            "crystalline_CeO2_BM31_protocol_Fm-3m": {
                "title": r"Input to model: [composition: None, spacegroup: $Fm\bar{3}m$]",
            },
            "crystalline_CeO2_BM31_protocol_none": {
                "title": r"Input to model: [composition: None, spacegroup: None]",
            },
        },
    },
    "s35_si_crystalline": {
        "pkl": "crystalline_Si.pkl",
        "output": "revision_main_si_crystalline.pdf",
        "config": {
            "Si_Mythen_protocol_Si4": {
                "title": r"Input to model: [composition: Si$_4$, spacegroup: None]",
                "zoom": 1.5,
            },
            "Si_Mythen_protocol_Si8": {
                "title": r"Input to model: [composition: Si$_8$, spacegroup: None]",
            },
            "Si_Mythen_protocol_none": {
                "title": r"Input to model: [composition: None, spacegroup: None]",
            },
            "Si_Mythen_protocol_Fd-3m": {
                "title": r"Input to model: [composition: None, spacegroup: $Fd\bar{3}m$]",
            },
            "Si_Mythen_protocol_Fm-3m": {
                "title": r"Input to model: [composition: None, spacegroup: $Fm\bar{3}m$]",
            },
            "Si_Mythen_protocol_Si4_Fm-3m": {
                "title": r"Input to model: [composition: Si$_4$, spacegroup: $Fm\bar{3}m$]",
            },
            "Si_Mythen_protocol_Si8_Fd-3m": {
                "title": r"Input to model: [composition: Si$_8$, spacegroup: $Fd\bar{3}m$]",
            },
        },
    },
    "s36_fe2o3_crystalline": {
        "pkl": "crystalline_Fe2O3.pkl",
        "output": "revision_main_fe2o3_crystalline.pdf",
        "config": {
            "AFS012d_a850C_protocol_Fe12O18": {
                "title": r"Input to model: [composition: Fe$_{12}$O$_{18}$, spacegroup: None]",
                "zoom": 1.4,
                "atom_offset_y": 2.0,
            },
            "AFS012d_a850C_protocol_Fe12O18_Ia-3": {
                "title": r"Input to model: [composition: Fe$_{12}$O$_{18}$, spacegroup: $Ia\bar{3}$] *",
                "zoom": 1.4,
                "atom_offset_y": 2.0,
            },
            "AFS012d_a850C_protocol_Fe12O18_Pna2_1": {
                "title": r"Input to model: [composition: Fe$_{12}$O$_{18}$, spacegroup: $Pna2_1$] *",
                "zoom": 1.4,
            },
            "AFS012d_a850C_protocol_Fe12O18_R-3c": {
                "title": r"Input to model: [composition: Fe$_{12}$O$_{18}$, spacegroup: $R\bar{3}c$]",
                "zoom": 1.4,
                "atom_offset_y": 2.0,
            },
            "AFS012d_a850C_protocol_Fe16O24": {
                "title": r"Input to model: [composition: Fe$_{16}$O$_{24}$, spacegroup: None]",
                "zoom": 1.5,
                "atom_offset_y": 2.0,
            },
            "AFS012d_a850C_protocol_Fe32O48": {
                "title": r"Input to model: [composition: Fe$_{32}$O$_{48}$, spacegroup: None]",
                "zoom": 1.5,
                "atom_offset_y": 2.0,
            },
            "AFS012d_a850C_protocol_Ia-3": {
                "title": r"Input to model: [composition: None, spacegroup: $Ia\bar{3}$]",
                "zoom": 1.4,
                "atom_offset_y": 2.0,
            },
            "AFS012d_a850C_protocol_Pna2_1": {
                "title": r"Input to model: [composition: None, spacegroup: $Pna2_1$]",
                "zoom": 1.4,
            },
            "AFS012d_a850C_protocol_R-3c": {
                "title": r"Input to model: [composition: None, spacegroup: $R\bar{3}c$]",
                "zoom": 1.3,
            },
            "AFS012d_a850C_protocol_none": {
                "title": r"Input to model: [composition: None, spacegroup: None]",
                "zoom": 1.4,
            },
        },
    },
    "s37_ceo2_particle": {
        "pkl": "particles_CeO2.pkl",
        "output": "revision_main_ceo2_particle.pdf",
        "config": {
            "Hydrolyse_ID5_20min_3-56_boro_0p8_protocol_Ce1O2": {
                "title": r"Input to model: [composition: Ce$_1$O$_2$, spacegroup: None]",
            },
            "Hydrolyse_ID5_20min_3-56_boro_0p8_protocol_Ce2O4": {
                "title": r"Input to model: [composition: Ce$_2$O$_4$, spacegroup: None]",
                "zoom": 1.35,
            },
            "Hydrolyse_ID5_20min_3-56_boro_0p8_protocol_Ce4O8": {
                "title": r"Input to model: [composition: Ce$_4$O$_8$, spacegroup: None]",
                "zoom": 1.35,
            },
            "Hydrolyse_ID5_20min_3-56_boro_0p8_protocol_Ce4O8_Fm-3m": {
                "title": r"Input to model: [composition: Ce$_4$O$_8$, spacegroup: $Fm\bar{3}m$]",
            },
            "Hydrolyse_ID5_20min_3-56_boro_0p8_protocol_Fm-3m": {
                "title": r"Input to model: [composition: None, spacegroup: $Fm\bar{3}m$]",
            },
            "Hydrolyse_ID5_20min_3-56_boro_0p8_protocol_none": {
                "title": r"Input to model: [composition: None, spacegroup: None]",
            },
        },
    },
}


OVERVIEW_SELECTION = [
    {
        "title": "(a)",
        "header": r"(a) Ce$_4$O$_8$ + $Fm\bar{3}m$",
        "source": "s34_ceo2_crystalline",
        "keys": [
            "crystalline_CeO2_BM31_protocol_none",
            "crystalline_CeO2_BM31_protocol_Ce4O8",
            "crystalline_CeO2_BM31_protocol_Ce4O8_Fm-3m",
        ],
    },
    {
        "title": "(b)",
        "header": r"(b) Si$_8$ + $Fd\bar{3}m$",
        "source": "s35_si_crystalline",
        "keys": [
            "Si_Mythen_protocol_none",
            "Si_Mythen_protocol_Si8",
            "Si_Mythen_protocol_Si8_Fd-3m",
        ],
    },
    {
        "title": "(c)",
        "header": r"(c) Fe$_{12}$O$_{18}$ + $R\bar{3}c$",
        "source": "s36_fe2o3_crystalline",
        "keys": [
            "AFS012d_a850C_protocol_none",
            "AFS012d_a850C_protocol_Fe12O18",
            "AFS012d_a850C_protocol_Fe12O18_R-3c",
        ],
    },
    {
        "title": "(d)",
        "header": r"(d) Ce$_4$O$_8$ + $Fm\bar{3}m$",
        "source": "s37_ceo2_particle",
        "keys": [
            "Hydrolyse_ID5_20min_3-56_boro_0p8_protocol_none",
            "Hydrolyse_ID5_20min_3-56_boro_0p8_protocol_Ce4O8",
            "Hydrolyse_ID5_20min_3-56_boro_0p8_protocol_Ce4O8_Fm-3m",
        ],
    },
]


def format_overview_title(title: str, *, prefix: str = "conditioning:") -> str:
    title = title.replace("Input to model: [", "")
    title = title.rstrip("]")
    match = re.match(r"composition:\s*(.*),\s*spacegroup:\s*(.*)", title)
    if match is None:
        return f"{prefix} {title}"
    composition = match.group(1).strip()
    spacegroup = match.group(2).strip()
    return f"{prefix} {composition} + {spacegroup}"


def select_rows_from_spec(spec_name: str, keys: list[str], *, title_prefix: str = "conditioning:") -> list[dict[str, Any]]:
    spec = APPENDIX_FIGURE_SPECS[spec_name]
    rows = build_rows_from_pickle(spec["pkl"], spec["config"])
    row_map = {key: row for key, row in zip(spec["config"].keys(), rows)}
    selected_rows = []
    for key in keys:
        row = dict(row_map[key])
        row["display_title"] = format_overview_title(row["title"], prefix=title_prefix)
        if spec_name == "s37_ceo2_particle":
            row["show_particles_fit"] = True
        if spec_name == "s36_fe2o3_crystalline":
            view = dict(row.get("view", {}))
            view["overview_zoom"] = 0.60
            row["view"] = view
        selected_rows.append(row)
    return selected_rows


def render_appendix_figures(output_dir: Path, selected: list[str] | None = None, *, dpi: int = 300) -> list[Path]:
    figure_names = selected or list(APPENDIX_FIGURE_SPECS.keys())
    outputs: list[Path] = []

    for name in figure_names:
        spec = APPENDIX_FIGURE_SPECS[name]
        output_path = output_dir / spec["output"]

        rows = build_rows_from_pickle(spec["pkl"], spec["config"])
        plot_pxrd_structures(rows, output_filename=output_path, dpi=dpi)
        outputs.append(output_path)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render appendix experimental figures from the shared pickles.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated figures. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--figure",
        action="append",
        choices=list(APPENDIX_FIGURE_SPECS.keys()),
        help="Render only the named figure. May be passed multiple times.",
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Render a lower-DPI draft for faster iteration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = render_appendix_figures(args.output_dir, args.figure, dpi=150 if args.draft else 300)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
