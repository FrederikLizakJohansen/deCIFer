import gzip
import importlib.util
import json
import os
import tempfile
import unittest

import pandas as pd
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter


MODULE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "bin", "plot_minicif_examples.py"
)
spec = importlib.util.spec_from_file_location("plot_minicif_examples", MODULE_PATH)
plot_minicif_examples = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_minicif_examples)


class PlotMinicifExamplesTest(unittest.TestCase):
    def test_selects_lowest_rwp_valid_candidate_per_sample(self):
        metrics = pd.DataFrame([
            {
                "split": "test",
                "sample_index": 1,
                "prompt_mode": "pxrd-elements",
                "reference_minicif": "reference",
                "generated_minicif": "generated-a",
                "structure_ok": True,
                "rwp": 0.4,
            },
            {
                "split": "test",
                "sample_index": 1,
                "prompt_mode": "pxrd-elements",
                "reference_minicif": "reference",
                "generated_minicif": "generated-b",
                "structure_ok": True,
                "rwp": 0.2,
            },
            {
                "split": "test",
                "sample_index": 2,
                "prompt_mode": "pxrd-elements",
                "reference_minicif": "reference",
                "generated_minicif": "invalid",
                "structure_ok": False,
                "rwp": 0.1,
            },
        ])

        selected = plot_minicif_examples.select_best_candidates(metrics)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.loc[0, "generated_minicif"], "generated-b")

    def test_refined_selection_requires_successful_refinement(self):
        metrics = pd.DataFrame([
            {
                "split": "test",
                "sample_index": 1,
                "prompt_mode": "pxrd",
                "reference_minicif": "reference",
                "generated_minicif": "failed-refinement",
                "structure_ok": True,
                "refinement_succeeded": False,
                "rep": 0,
                "rwp": 0.1,
            },
            {
                "split": "test",
                "sample_index": 1,
                "prompt_mode": "pxrd",
                "reference_minicif": "reference",
                "generated_minicif": "successful-refinement",
                "structure_ok": True,
                "refinement_succeeded": True,
                "rep": 1,
                "rwp": 0.2,
            },
        ])

        selected = plot_minicif_examples.select_best_candidates(
            metrics,
            require_refined=True,
        )

        self.assertEqual(
            selected.loc[0, "generated_minicif"],
            "successful-refinement",
        )

    def test_plots_structures_from_saved_evaluation_rows(self):
        reference = (
            "<mcif> Na cs_7 sg_221 cell "
            "3.000 3.000 3.000 90.000 90.000 90.000 "
            "<atom> Na 1 0.000 0.000 0.000 1.000 </mcif>"
        )
        generated = (
            "<mcif> Na cs_7 sg_221 cell "
            "3.100 3.100 3.100 90.000 90.000 90.000 "
            "<atom> Na 1 0.000 0.000 0.000 1.000 </mcif>"
        )
        candidates = pd.DataFrame([{
            "split": "test",
            "sample_index": 3,
            "prompt_mode": "pxrd-elements",
            "cif_name": "toy",
            "rep": 0,
            "rwp": 0.3,
            "reference_minicif": reference,
            "generated_minicif": generated,
        }])
        xrd_kwargs = {
            "qmin": 0.0,
            "qmax": 4.0,
            "qstep": 0.02,
            "fwhm_range": [0.05, 0.05],
            "eta_range": [0.5, 0.5],
            "noise_range": None,
            "intensity_scale_range": None,
            "mask_prob": None,
            "final_normalize": True,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = plot_minicif_examples.plot_examples(
                candidates, xrd_kwargs, tmpdir, "CuKa", 1
            )
            figure_path = os.path.join(
                tmpdir,
                "test",
                "sample_0000003_pxrd-elements.png",
            )

            self.assertTrue(os.path.isfile(figure_path))
            self.assertTrue(os.path.isfile(manifest_path))
            self.assertEqual(len(pd.read_csv(manifest_path)), 1)

    def test_loads_and_plots_saved_refined_structure(self):
        reference = (
            "<mcif> Na cs_7 sg_221 cell "
            "3.000 3.000 3.000 90.000 90.000 90.000 "
            "<atom> Na 1 0.000 0.000 0.000 1.000 </mcif>"
        )
        generated = (
            "<mcif> Na cs_7 sg_221 cell "
            "3.100 3.100 3.100 90.000 90.000 90.000 "
            "<atom> Na 1 0.000 0.000 0.000 1.000 </mcif>"
        )
        candidates = pd.DataFrame([{
            "split": "test",
            "sample_index": 3,
            "prompt_mode": "pxrd-elements",
            "cif_name": "toy",
            "rep": 2,
            "rwp": 0.3,
            "refined_rwp": 0.1,
            "reference_minicif": reference,
            "generated_minicif": generated,
        }])
        refined_structure = Structure(
            Lattice.cubic(3.02),
            ["Na"],
            [[0, 0, 0]],
        )
        record = {
            "split": "test",
            "sample_index": 3,
            "prompt_mode": "pxrd-elements",
            "rep": 2,
            "succeeded": True,
            "status": "converged",
            "refined_cif": str(CifWriter(refined_structure, symprec=None)),
        }
        xrd_kwargs = {
            "qmin": 0.0,
            "qmax": 4.0,
            "qstep": 0.02,
            "fwhm_range": [0.05, 0.05],
            "eta_range": [0.5, 0.5],
            "noise_range": None,
            "intensity_scale_range": None,
            "mask_prob": None,
            "final_normalize": True,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = os.path.join(
                tmpdir,
                "minicif_refinement_results.jsonl.gz",
            )
            with gzip.open(archive_path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            refinements = plot_minicif_examples.load_saved_refinements(
                tmpdir,
                candidates,
            )
            manifest_path = plot_minicif_examples.plot_examples(
                candidates,
                xrd_kwargs,
                os.path.join(tmpdir, "figures"),
                "CuKa",
                1,
                refinements,
            )
            manifest = pd.read_csv(manifest_path)

            self.assertEqual(len(refinements), 1)
            self.assertEqual(manifest.loc[0, "refinement_status"], "converged")
            self.assertTrue(
                os.path.isfile(manifest.loc[0, "figure_path"])
            )
            self.assertLess(
                manifest.loc[0, "rendered_refined_rwp"],
                manifest.loc[0, "rendered_rwp"],
            )

    def test_default_selection_can_cover_every_crystal_system(self):
        candidates = pd.DataFrame([
            {
                "sample_index": crystal_system,
                "reference_crystal_system": crystal_system,
                "rwp": 0.1 * crystal_system,
            }
            for crystal_system in range(1, 8)
            for _ in range(2)
        ])

        selected = plot_minicif_examples.choose_examples(
            candidates,
            count=7,
            selection="crystal-system",
            seed=1337,
        )

        self.assertEqual(len(selected), 7)
        self.assertEqual(set(selected["reference_crystal_system"]), set(range(1, 8)))


if __name__ == "__main__":
    unittest.main()
