#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from analysis_figure_exports import (
    load_appendix_ablation_datasets,
    load_main_ablation_datasets,
    run_main_robustness_figure,
    run_summary_robustness_figures,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate robustness and summary figure exports.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--section",
        nargs="+",
        choices=["main", "summaries", "all"],
        default=["all"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sections = set(args.section)
    if "all" in sections:
        sections = {"main", "summaries"}

    if "main" in sections:
        run_main_robustness_figure(args.output_dir / "robustness", load_main_ablation_datasets())
    if "summaries" in sections:
        run_summary_robustness_figures(args.output_dir / "summaries", load_appendix_ablation_datasets())


if __name__ == "__main__":
    main()
