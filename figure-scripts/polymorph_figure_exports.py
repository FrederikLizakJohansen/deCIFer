#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from analysis_figure_exports import load_polymorph_datasets, run_polymorph_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FeO2 polymorph figure exports.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_polymorph_figure(args.output_dir / "polymorphs", load_polymorph_datasets())


if __name__ == "__main__":
    main()
