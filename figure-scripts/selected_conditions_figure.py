#!/usr/bin/env python3

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase


REPO_ROOT = Path(__file__).resolve().parent
SOURCE_PATH = REPO_ROOT / "experimental_appendix_figures.py"
DEFAULT_OUTPUT = REPO_ROOT / "final-figures" / "revision" / "selected_conditions_updated.pdf"


class StemLegendHandle:
    pass


class StemLegendHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        color = orig_handle.color
        cx = xdescent + 0.5 * width
        y0 = ydescent + 0.15 * height
        y1 = ydescent + 0.82 * height
        stem = Line2D([cx, cx], [y0, y1], color=color, linewidth=1.0, transform=trans)
        marker = Line2D(
            [cx],
            [y1],
            color=color,
            marker="o",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.0,
            linestyle="None",
            transform=trans,
        )
        return [stem, marker]


def add_local_pattern_legend(ax, revision, *, include_scherrer=False):
    stem_handle = StemLegendHandle()
    stem_handle.color = revision.color_prediction_stem
    handles = [
        Line2D([0], [0], color=revision.color_data, lw=1.2, label="data"),
        stem_handle,
    ]
    labels = ["data", "best prediction"]
    if include_scherrer:
        handles.append(
            Line2D(
                [0],
                [0],
                color=revision.color_particles_fit,
                lw=1.45,
                linestyle=(0, (1.0, 1.8)),
                label=rf"Scherrer, $\tau = {int(revision.scherrer_tau_angstrom)}$ $\AA$",
            )
        )
        labels.append(rf"Scherrer, $\tau = {int(revision.scherrer_tau_angstrom)}$ $\AA$")
    ax.legend(
        handles=handles,
        labels=labels,
        loc="lower left",
        bbox_to_anchor=(0.11, 1.13),
        ncol=3 if include_scherrer else 2,
        frameon=False,
        fontsize=8.6,
        handlelength=1.8,
        columnspacing=0.9,
        borderaxespad=0.0,
        handler_map={StemLegendHandle: StemLegendHandler()},
    )


def load_revision_module():
    spec = importlib.util.spec_from_file_location("appendix_figure_source", SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def draw_row_pair_large(
    revision,
    fig,
    left_spec,
    right_spec,
    row,
    *,
    bottom_row,
    block_header=None,
    block_ref_label=None,
    row_sublabel=None,
):
    ax_pxrd = fig.add_subplot(left_spec)
    ax_struc = fig.add_subplot(right_spec)
    ax_struc.axis("off")

    title = row.get("display_title", row["title"])
    header_formula = None
    header_spacegroup = None
    if block_header is not None:
        panel_label = None
        header_text = block_header.replace("ref: ", "")
        if ") " in header_text:
            panel_label, header_text = header_text.split(") ", 1)
            panel_label = f"{panel_label})"
        if " + " in header_text:
            header_formula, header_spacegroup = header_text.split(" + ", 1)
        else:
            header_formula, header_spacegroup = header_text, ""
        if panel_label is not None:
            ax_pxrd.text(
                0.0,
                1.20,
                panel_label,
                transform=ax_pxrd.transAxes,
                ha="left",
                va="bottom",
                fontsize=11.6,
                fontweight="normal",
                color="black",
            )

    if row_sublabel is not None:
        ax_pxrd.text(
            0.012,
            0.95,
            row_sublabel,
            transform=ax_pxrd.transAxes,
            ha="left",
            va="top",
            fontsize=10.4,
            fontweight="normal",
            color="black",
            zorder=6,
        )

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
        color=revision.color_prediction_stem,
        linestyle=revision.stem_linestyle,
        linewidth=revision.stem_linewidth,
        zorder=2,
    )
    plt.setp(
        stem.markerline,
        markersize=revision.marker_size,
        markerfacecolor="white",
        markeredgecolor=revision.color_prediction_stem,
        markeredgewidth=1.0,
        zorder=2,
    )
    plt.setp(stem.baseline, zorder=2)

    ax_pxrd.plot(row["q"], row["iq"], lw=1.15, c=revision.color_data, alpha=0.92, zorder=3, label="data")
    if row.get("show_particles_fit"):
        iq_scherrer = revision.scherrer_broaden_in_q(row["q"], row["q_disc"], row["iq_disc"])
        ax_pxrd.plot(
            row["q"],
            iq_scherrer,
            lw=1.45,
            c=revision.color_particles_fit,
            alpha=1.0,
            linestyle=(0, (1.0, 1.8)),
            zorder=4,
        )

    ax_pxrd.set(xlim=(1.5, 7.5), ylim=(0, None), yticklabels=[], yticks=[])
    ax_pxrd.grid(alpha=0.28, axis="both", linestyle="-", linewidth=0.55)
    if bottom_row:
        ax_pxrd.set_xlabel(r"$Q_{\;[\AA^{-1}]}$", fontsize=14, labelpad=2)
    else:
        ax_pxrd.tick_params(axis="x", labelbottom=False)
    ax_pxrd.tick_params(axis="both", labelsize=13)

    revision.plot_unit_cell_with_boundaries(row["structure"], ax=ax_struc, radii=0.5)
    xlim = ax_struc.get_xlim()
    ylim = ax_struc.get_ylim()
    xpad = (xlim[1] - xlim[0]) * 0.10
    ypad_bottom = (ylim[1] - ylim[0]) * 0.15
    ypad_top = (ylim[1] - ylim[0]) * 0.09
    ax_struc.set_xlim((xlim[0] - xpad, xlim[1] + xpad))
    ax_struc.set_ylim((ylim[0] - ypad_bottom, ylim[1] + ypad_top))

    view = row.get("view", {})
    revision.adjust_unit_cell_view(
        ax_struc,
        zoom=view.get("overview_zoom", view.get("zoom", 1.0) * 0.66),
        dx=view.get("dx", 0.0),
        dy=view.get("dy", 0.0),
    )
    revision.disable_axis_clipping(ax_struc)

    pred_label = f"{row['formula_latex']}\n{row['spacegroup_latex']}"
    if block_ref_label is not None:
        ax_pxrd.text(
            0.80,
            0.84,
            block_ref_label,
            transform=ax_pxrd.transAxes,
            ha="right",
            va="top",
            fontsize=10.2,
            linespacing=1.02,
            color="black",
            zorder=6,
        )
    ax_pxrd.text(
        0.965,
        0.84,
        pred_label,
        transform=ax_pxrd.transAxes,
        ha="right",
        va="top",
        fontsize=10.2,
        linespacing=0.92,
        color=revision.color_prediction_label,
        bbox=dict(facecolor="white", alpha=0.64, edgecolor="none", pad=0.45),
    )

    return ax_pxrd, ax_struc


def plot_compact_statistics_style(revision, grouped_rows, *, output_filename, dpi=300):
    nblocks = len(grouped_rows)
    ncols = 2
    nrows = (nblocks + ncols - 1) // ncols
    fig = plt.figure(figsize=(15.7, 7.0), dpi=dpi)
    outer = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=-0.11, hspace=0.38)

    block_axes = []
    has_particles_fit = False

    for block_idx, block in enumerate(grouped_rows):
        header_text = block["header"].replace("ref: ", "")
        if ") " in header_text:
            _, header_text = header_text.split(") ", 1)
        if " + " in header_text:
            block_ref_label = "\n".join(header_text.split(" + ", 1))
        else:
            block_ref_label = header_text
        sub = gridspec.GridSpecFromSubplotSpec(
            len(block["rows"]),
            2,
            subplot_spec=outer[block_idx // ncols, block_idx % ncols],
            width_ratios=[0.94, 1.06],
            wspace=-0.36,
            hspace=0.18,
        )

        top_ax = None
        bottom_ax = None
        for row_idx, row in enumerate(block["rows"]):
            sublabel = ["(i)", "(ii)", "(iii)"][row_idx] if row_idx < 3 else None
            ax_pxrd, _ = draw_row_pair_large(
                revision,
                fig,
                sub[row_idx, 0],
                sub[row_idx, 1],
                row,
                bottom_row=(row_idx == len(block["rows"]) - 1),
            block_header=block["header"] if row_idx == 0 else None,
                block_ref_label=block_ref_label,
                row_sublabel=sublabel,
            )
            if row_idx == 0:
                add_local_pattern_legend(
                    ax_pxrd,
                    revision,
                    include_scherrer=row.get("show_particles_fit", False),
                )
            has_particles_fit = has_particles_fit or row.get("show_particles_fit", False)
            if row_idx == 0:
                top_ax = ax_pxrd
            bottom_ax = ax_pxrd
        if top_ax is not None and bottom_ax is not None:
            block_axes.append((top_ax, bottom_ax))

    for top_ax, bottom_ax in block_axes:
        top_pos = top_ax.get_position()
        bottom_pos = bottom_ax.get_position()
        fig.text(
            top_pos.x0 - 0.014,
            (top_pos.y1 + bottom_pos.y0) / 2,
            r"$I(Q)$ [a.u.]",
            va="center",
            ha="center",
            rotation="vertical",
            fontsize=14,
        )

    revision.save_figure_outputs(fig, output_filename)
    plt.close(fig)


def build_grouped_rows(revision):
    grouped_rows = []
    for block in revision.OVERVIEW_SELECTION:
        header = block["header"]
        if ") " in header:
            prefix, rest = header.split(") ", 1)
            header = f"{prefix}) ref: {rest}"
        else:
            header = f"ref: {header}"
        grouped_rows.append(
            {
                "title": block["title"],
                "header": header,
                "rows": revision.select_rows_from_spec(
                    block["source"],
                    block["keys"],
                    title_prefix="cond:",
                ),
            }
        )
    return grouped_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render the updated selected-conditions experimental figure."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PDF path. PNG is written alongside it. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Render a lower-DPI draft for faster iteration.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    revision = load_revision_module()
    grouped_rows = build_grouped_rows(revision)
    plot_compact_statistics_style(
        revision,
        grouped_rows,
        output_filename=args.output,
        dpi=150 if args.draft else 300,
    )
    print(args.output)


if __name__ == "__main__":
    main()
