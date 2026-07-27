import importlib.util
import os
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

from decifer.minicif import MinicifTokenizer
from decifer.minicif_v2 import MinicifV2Tokenizer

MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "bin", "visualize_minicif.py")
spec = importlib.util.spec_from_file_location("visualize_minicif", MODULE_PATH)
visualize_minicif = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visualize_minicif)
prompt_from_minicif = visualize_minicif.prompt_from_minicif
summarize = visualize_minicif.summarize
compatible_evaluation_indices = visualize_minicif.compatible_evaluation_indices


class VisualizeMinicifTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
