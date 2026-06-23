#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "revision-final-figures" / "generated"


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    python = sys.executable

    run(
        [
            python,
            str(FIGURE_DIR / "experimental_appendix_figures.py"),
            "--figure",
            "s34_ceo2_crystalline",
            "--figure",
            "s35_si_crystalline",
            "--figure",
            "s36_fe2o3_crystalline",
            "--figure",
            "s37_ceo2_particle",
            "--output-dir",
            str(OUTPUT_ROOT / "experimental"),
        ]
    )
    run(
        [
            python,
            str(FIGURE_DIR / "selected_conditions_figure.py"),
            "--output",
            str(
                OUTPUT_ROOT
                / "experimental"
                / "selected_conditions_updated.pdf"
            ),
        ]
    )
    run(
        [
            python,
            str(FIGURE_DIR / "selected_conditions_v2.py"),
            "--output",
            str(
                OUTPUT_ROOT
                / "experimental"
                / "selected_conditions_v2.pdf"
            ),
        ]
    )
    run(
        [
            python,
            str(FIGURE_DIR / "experimental_results_figure.py"),
            "--output-dir",
            str(OUTPUT_ROOT / "experimental"),
            "--filename",
            "experimental_results.pdf",
        ]
    )
    run(
        [
            python,
            str(FIGURE_DIR / "rwp_ranking_sensitivity.py"),
            "--mode",
            "plot",
            "--profile",
            "paper_fe_tau",
            "--plot-style",
            "paper",
            "--input-dir",
            str(REPO_ROOT / "final-figures" / "revision" / "ranking-sensitivity-paper-fe-tau"),
            "--output-dir",
            str(OUTPUT_ROOT / "ranking-sensitivity"),
        ]
    )
    run(
        [
            python,
            str(FIGURE_DIR / "ablation_figure_exports.py"),
            "--output-dir",
            str(OUTPUT_ROOT / "analysis"),
            "--section",
            "all",
        ]
    )
    run(
        [
            python,
            str(FIGURE_DIR / "robustness_figure_exports.py"),
            "--output-dir",
            str(OUTPUT_ROOT / "analysis"),
            "--section",
            "all",
        ]
    )
    run(
        [
            python,
            str(FIGURE_DIR / "polymorph_figure_exports.py"),
            "--output-dir",
            str(OUTPUT_ROOT / "analysis"),
        ]
    )


if __name__ == "__main__":
    main()
