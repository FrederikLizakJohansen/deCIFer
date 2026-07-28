import importlib.util
import os
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure

from decifer.minicif import MinicifTokenizer
from decifer.minicif_v2 import MinicifV2Tokenizer

MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "bin", "visualize_minicif.py")
spec = importlib.util.spec_from_file_location("visualize_minicif", MODULE_PATH)
visualize_minicif = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visualize_minicif)
prompt_from_minicif = visualize_minicif.prompt_from_minicif
summarize = visualize_minicif.summarize
compatible_evaluation_indices = visualize_minicif.compatible_evaluation_indices
plot_rwp_distribution = visualize_minicif.plot_rwp_distribution
save_evaluation_example = visualize_minicif.save_evaluation_example


class VisualizeMinicifTest(unittest.TestCase):
    def test_save_evaluation_example_writes_split_specific_figure(self):
        structure = Structure(Lattice.cubic(3.0), ["Na"], [[0, 0, 0]])
        reference_iq = np.asarray([1.0, 0.5, 0.0])
        rows = [{
            "rep": 0,
            "rwp": 0.1,
            "generated_iq": np.asarray([0.9, 0.4, 0.0]),
            "generated_structure": structure,
        }]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_evaluation_example(
                tmpdir,
                "test",
                42,
                "pxrd-elements",
                {"qmin": 0.0, "qstep": 0.1},
                reference_iq,
                structure,
                rows,
                "toy",
                1,
            )

            self.assertTrue(
                path.endswith(
                    "examples/test/sample_0000042_pxrd-elements.png"
                )
            )
            self.assertTrue(os.path.isfile(path))

    def test_evaluation_excludes_references_beyond_checkpoint_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.h5")
            with h5py.File(path, "w") as h5:
                h5.create_dataset(
                    "cif_token_length",
                    data=np.asarray([100, 641, 642], dtype=np.int32),
                )

            indices = compatible_evaluation_indices(
                path,
                {
                    "block_size": 640,
                    "condition": True,
                    "condition_cross_attention": True,
                },
            )

        np.testing.assert_array_equal(indices, [0, 1])

    def test_pxrd_prompt_modes_force_known_fields(self):
        tokenizer = MinicifTokenizer()
        minicif = (
            "<mcif> Na Cl cs_7 sg_221 cell "
            "5.640 5.640 5.640 90.000 90.000 90.000 "
            "<atom> Na 4 0.000 0.000 0.000 1.000 </mcif>"
        )

        expected = {
            "pxrd": "<mcif>",
            "pxrd-elements": "<mcif> Na Cl",
            "pxrd-elements-cs": "<mcif> Na Cl cs_7",
            "pxrd-elements-cs-sg": "<mcif> Na Cl cs_7 sg_221",
        }

        for mode, prompt in expected.items():
            ids = prompt_from_minicif(minicif, mode, tokenizer)
            self.assertEqual(tokenizer.decode(ids.tolist()), prompt)

    def test_legacy_prompt_mode_names_are_still_supported(self):
        tokenizer = MinicifTokenizer()
        minicif = "<mcif> Na Cl cs_7 sg_221 cell 1.000 1.000 1.000 90.000 90.000 90.000 </mcif>"

        self.assertEqual(
            tokenizer.decode(prompt_from_minicif(minicif, "start", tokenizer).tolist()),
            tokenizer.decode(prompt_from_minicif(minicif, "pxrd", tokenizer).tolist()),
        )
        self.assertEqual(
            tokenizer.decode(prompt_from_minicif(minicif, "formula-cs-sg", tokenizer).tolist()),
            tokenizer.decode(prompt_from_minicif(minicif, "pxrd-elements-cs-sg", tokenizer).tolist()),
        )

    def test_v2_stoichiometry_is_optional_in_prompt(self):
        tokenizer = MinicifV2Tokenizer()
        minicif = (
            "<mcif2> Na Cl formula Na 1 Cl 1 cs_7 sg_225 cell "
            "5.6400 5.6400 5.6400 90.0000 90.0000 90.0000 "
            "<atom> Na wp_a 0.0000 0.0000 0.0000 1.0000 </mcif2>"
        )

        expected = {
            "pxrd": "<mcif2>",
            "pxrd-elements": "<mcif2> Na Cl formula",
            "pxrd-stoichiometry": "<mcif2> Na Cl formula Na 1 Cl 1",
            "pxrd-stoichiometry-cs": "<mcif2> Na Cl formula Na 1 Cl 1 cs_7",
            "pxrd-stoichiometry-cs-sg": "<mcif2> Na Cl formula Na 1 Cl 1 cs_7 sg_225",
        }

        for mode, prompt in expected.items():
            ids = prompt_from_minicif(minicif, mode, tokenizer)
            self.assertEqual(tokenizer.decode(ids.tolist()), prompt)

    def test_summary_includes_element_set_and_structure_rates(self):
        df = pd.DataFrame([
            {
                "split": "val",
                "sample_index": 0,
                "rep": 0,
                "parse_ok": True,
                "structure_ok": True,
                "match": False,
                "rwp": 0.4,
                "space_group_match": False,
                "crystal_system_match": True,
                "element_set_match": True,
                "composition_match": False,
                "formula_match": True,
            },
            {
                "split": "val",
                "sample_index": 0,
                "rep": 1,
                "parse_ok": False,
                "structure_ok": False,
                "match": False,
                "space_group_match": False,
                "crystal_system_match": False,
                "element_set_match": False,
                "composition_match": False,
                "formula_match": False,
            },
        ])

        summary = summarize(df)

        self.assertEqual(summary.loc[0, "valid_minicif_rate"], 0.5)
        self.assertEqual(summary.loc[0, "structure_rate"], 0.5)
        self.assertEqual(summary.loc[0, "element_set_accuracy"], 0.5)
        self.assertEqual(summary.loc[0, "formula_accuracy"], 0.5)

    def test_plot_rwp_distribution_writes_report(self):
        df = pd.DataFrame(
            {
                "split": ["test", "test"],
                "rwp": [0.2, 0.4],
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_rwp_distribution(df, tmpdir)
            self.assertTrue(
                os.path.isfile(os.path.join(tmpdir, "rwp_distribution.png"))
            )


if __name__ == "__main__":
    unittest.main()
