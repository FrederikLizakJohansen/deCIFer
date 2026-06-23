#!/usr/bin/env python3
"""Updated selected-conditions experimental figure (v2).

Redesign of selected_conditions_figure.py for the revision: the figure is laid
out at its true print size (0.9 * linewidth on A4 with 3 cm margins, i.e. about
5.3 in wide) so that every font size in this script is the size the reader sees.
Four material blocks (a-d) in a 2x2 grid, each with three conditioning rows of
PXRD + predicted unit cell. Reference labels live in the block headers, one
shared legend sits on top, and the structure panels get roughly three times the
print area of the previous version.
"""

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SOURCE_PATH = HERE.parent / "experimental_appendix_figures.py"
DEFAULT_OUTPUT = HERE / "generated" / "selected_conditions_v2.pdf"

# All sizes below are the printed sizes (figure is included at ~1:1 scale).
FIG_WIDTH = 5.31
FIG_HEIGHT = 7.05

FS_TICK = 7.0
FS_AXIS_LABEL = 8.5
FS_HEADER = 9.0
FS_HEADER_REF = 7.5
FS_SUBLABEL = 7.5
FS_PRED_LABEL = 7.5
FS_LEGEND = 8.0
FS_ATOM_LEGEND = 6.0

DATA_LW = 0.8
STEM_LW = 0.8
STEM_MARKERSIZE = 2.6
SCHERRER_LW = 1.0

BLOCK_HEADERS = {
    "(a)": (r"CeO$_2$", r"ref: Ce$_4$O$_8$, $Fm\bar{3}m$"),
    "(b)": (r"Si", r"ref: Si$_8$, $Fd\bar{3}m$"),
    "(c)": (r"Fe$_2$O$_3$", r"ref: Fe$_{12}$O$_{18}$, $R\bar{3}c$"),
    "(d)": (r"nano-CeO$_2$", r"ref: Ce$_4$O$_8$, $Fm\bar{3}m$"),
}

# Fraction of the structure-axes height reserved at the bottom for the atom key.
ATOM_KEY_STRIP = 0.14


def load_revision_module():
    spec = importlib.util.spec_from_file_location("appendix_figure_source", SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class StemLegendHandle:
    def __init__(self, color):
        self.color = color


class StemLegendHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        color = orig_handle.color
        cx = xdescent + 0.5 * width
        y0 = ydescent + 0.10 * height
        y1 = ydescent + 0.80 * height
        stem = Line2D([cx, cx], [y0, y1], color=color, linewidth=STEM_LW, transform=trans)
        marker = Line2D(
            [cx], [y1],
            color=color, marker="o", markersize=4.5,
            markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.8,
            linestyle="None", transform=trans,
        )
        return [stem, marker]


def add_global_legend(fig, revision):
    stem_handle = StemLegendHandle(revision.color_prediction_stem)
    handles = [
        Line2D([0], [0], color=revision.color_data, lw=DATA_LW),
        stem_handle,
        Line2D(
            [0], [0],
            color=revision.color_particles_fit, lw=SCHERRER_LW, linestyle=(0, (1.0, 1.6)),
        ),
    ]
    labels = [
        "data",
        "best prediction",
        rf"Scherrer, $\tau = {int(revision.scherrer_tau_angstrom)}\,\AA$ (d only)",
    ]
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        fontsize=FS_LEGEND,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.4,
        borderaxespad=0.0,
        handler_map={StemLegendHandle: StemLegendHandler()},
    )


def draw_block_header(fig, header_axes, panel_label):
    """Header line spanning the block: '(a) material' left, 'ref: ...' right."""
    left_ax, right_ax = header_axes
    material, ref = BLOCK_HEADERS[panel_label]
    left_pos = left_ax.get_position()
    right_pos = right_ax.get_position()
    y = left_pos.y1 + 0.012
    fig.text(
        left_pos.x0, y,
        f"{panel_label}  {material}",
        ha="left", va="bottom", fontsize=FS_HEADER, color="black",
    )
    fig.text(
        right_pos.x1, y,
        ref,
        ha="right", va="bottom", fontsize=FS_HEADER_REF, color="0.25",
    )


def fit_structure_view(ax, *, zoom=1.0, dx_frac=0.0):
    """Pad the auto-fitted limits: small margins on the sides and top, plus a
    reserved strip below the structure for the atom key."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xspan = (x1 - x0) * zoom
    yspan = (y1 - y0) * zoom
    xc = (x0 + x1) / 2 + dx_frac * xspan
    ax.set_xlim(xc - xspan / 2 * 1.10, xc + xspan / 2 * 1.10)
    ax.set_ylim(y0 - (ATOM_KEY_STRIP + 0.04) * yspan, y1 + 0.10 * yspan)


def draw_atom_legend(revision, ax_struc, structure):
    """Small element key (colored circles + symbols) under the structure."""
    species = sorted({site.species_string for site in structure})
    x0, x1 = ax_struc.get_xlim()
    y0, y1 = ax_struc.get_ylim()
    span = x1 - x0
    spacing = 0.22 * span
    start = (x0 + x1) / 2 - spacing * (len(species) - 1) / 2
    y = y0 + 0.08 * (y1 - y0)
    radius_px = 5.0 * ax_struc.figure.dpi / 72.0
    for i, sp in enumerate(species):
        x = start + i * spacing
        atom_color = revision.get_atom_color(sp)
        circ = revision.fixed_size_circle(
            x, y, radius_points=radius_px, ax=ax_struc,
            edgecolor="black", facecolor=atom_color,
            lw=0.6, zorder=10, clip_on=False,
        )
        ax_struc.add_patch(circ)
        r, g, b = mcolors.to_rgb(atom_color)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        ax_struc.text(
            x, y, sp, color="white" if luminance < 0.5 else "black",
            ha="center", va="center",
            fontsize=FS_ATOM_LEGEND, zorder=11, clip_on=False,
        )


def draw_row(revision, fig, left_spec, right_spec, row, *, bottom_row, row_sublabel):
    ax_pxrd = fig.add_subplot(left_spec)
    ax_struc = fig.add_subplot(right_spec)
    ax_struc.axis("off")

    stem = ax_pxrd.stem(
        row["q_disc"], row["iq_disc"],
        linefmt="-", markerfmt="o", basefmt=" ",
    )
    plt.setp(stem.stemlines, color=revision.color_prediction_stem,
             linestyle="-", linewidth=STEM_LW, zorder=2)
    plt.setp(stem.markerline, markersize=STEM_MARKERSIZE, markerfacecolor="white",
             markeredgecolor=revision.color_prediction_stem, markeredgewidth=0.8, zorder=2)
    plt.setp(stem.baseline, zorder=2)

    ax_pxrd.plot(row["q"], row["iq"], lw=DATA_LW, c=revision.color_data,
                 alpha=0.92, zorder=3)

    if row.get("show_particles_fit"):
        iq_scherrer = revision.scherrer_broaden_in_q(row["q"], row["q_disc"], row["iq_disc"])
        ax_pxrd.plot(row["q"], iq_scherrer, lw=SCHERRER_LW,
                     c=revision.color_particles_fit, linestyle=(0, (1.0, 1.6)), zorder=4)

    ax_pxrd.set(xlim=(1.5, 7.5), ylim=(0, 1.18), yticks=[])
    ax_pxrd.set_xticks([2, 3, 4, 5, 6, 7])
    ax_pxrd.grid(alpha=0.25, axis="x", linestyle="-", linewidth=0.4)
    if bottom_row:
        ax_pxrd.set_xlabel(r"$Q$ [$\AA^{-1}$]", fontsize=FS_AXIS_LABEL, labelpad=1.5)
    else:
        ax_pxrd.tick_params(axis="x", labelbottom=False)
    ax_pxrd.tick_params(axis="both", labelsize=FS_TICK, length=2.2, pad=1.5)
    for spine in ax_pxrd.spines.values():
        spine.set_linewidth(0.6)

    ax_pxrd.text(
        0.025, 0.955, row_sublabel,
        transform=ax_pxrd.transAxes, ha="left", va="top",
        fontsize=FS_SUBLABEL, color="black", zorder=6,
    )

    pred_label = f"{row['formula_latex']}\n{row['spacegroup_latex']}"
    ax_pxrd.text(
        0.97, 0.95, pred_label,
        transform=ax_pxrd.transAxes, ha="right", va="top",
        fontsize=FS_PRED_LABEL, linespacing=1.0,
        color=revision.color_prediction_label, zorder=6,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.4),
    )

    # Structure panel
    revision.plot_unit_cell_with_boundaries(row["structure"], ax=ax_struc, radii=0.5)
    struc_view = row.get("v2_view", {})
    fit_structure_view(
        ax_struc,
        zoom=struc_view.get("zoom", 1.15),
        dx_frac=struc_view.get("dx_frac", 0.0),
    )
    draw_atom_legend(revision, ax_struc, row["structure"])

    return ax_pxrd, ax_struc


# Per-row structure view tweaks, keyed by (panel label, row index).
V2_VIEWS = {}


def fix_spacegroup_bar(latex):
    """The shared converter puts the bar over all trailing digits (F-43m ->
    F bar{43} m); restrict it to the first digit only."""
    return re.sub(r"\\bar\{(\d)(\d+)\}", r"\\bar{\1}\2", latex)


def build_grouped_rows(revision):
    grouped_rows = []
    for block in revision.OVERVIEW_SELECTION:
        rows = revision.select_rows_from_spec(block["source"], block["keys"], title_prefix="cond:")
        for row_idx, row in enumerate(rows):
            row["v2_view"] = V2_VIEWS.get((block["title"], row_idx), {})
            row["spacegroup_latex"] = fix_spacegroup_bar(row["spacegroup_latex"])
        grouped_rows.append({"title": block["title"], "rows": rows})
    return grouped_rows


def plot_figure(revision, grouped_rows, *, output_filename, dpi=300):
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=dpi)
    outer = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.045, right=0.985, top=0.925, bottom=0.045,
        wspace=0.16, hspace=0.30,
    )

    for block_idx, block in enumerate(grouped_rows):
        sub = gridspec.GridSpecFromSubplotSpec(
            3, 2,
            subplot_spec=outer[block_idx // 2, block_idx % 2],
            width_ratios=[1.45, 1.0],
            wspace=0.04,
            hspace=0.14,
        )
        header_axes = None
        for row_idx, row in enumerate(block["rows"]):
            ax_pxrd, ax_struc = draw_row(
                revision, fig, sub[row_idx, 0], sub[row_idx, 1], row,
                bottom_row=(row_idx == 2),
                row_sublabel=["(i)", "(ii)", "(iii)"][row_idx],
            )
            if row_idx == 0:
                header_axes = (ax_pxrd, ax_struc)
            if row_idx == 1:
                ax_pxrd.set_ylabel(r"$I(Q)$ [a.u.]", fontsize=FS_AXIS_LABEL, labelpad=2)
        draw_block_header(fig, header_axes, block["title"])

    add_global_legend(fig, revision)

    output_filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_filename, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_filename.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render the v2 selected-conditions experimental figure at print size."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output PDF path (PNG written alongside). Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--draft", action="store_true",
                        help="Render a lower-DPI draft for faster iteration.")
    return parser.parse_args()


def main():
    args = parse_args()
    revision = load_revision_module()
    grouped_rows = build_grouped_rows(revision)
    plot_figure(revision, grouped_rows, output_filename=args.output,
                dpi=150 if args.draft else 300)
    print(args.output)


if __name__ == "__main__":
    main()
