#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "bin") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "bin"))

from analysis_figure_exports import (
    load_appendix_ablation_datasets,
    load_main_ablation_datasets,
    run_appendix_ablation_figures,
    run_main_ablation_figures,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ablation figure exports.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--section",
        nargs="+",
        choices=["main", "appendix", "all"],
        default=["all"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sections = set(args.section)
    if "all" in sections:
        sections = {"main", "appendix"}

    if "main" in sections:
        run_main_ablation_figures(args.output_dir / "main-ablation", load_main_ablation_datasets())
    if "appendix" in sections:
        run_appendix_ablation_figures(args.output_dir / "appendix-ablation", load_appendix_ablation_datasets())


if __name__ == "__main__":
    main()
