import importlib.util
import os
import unittest

import pandas as pd

from decifer.minicif import MinicifTokenizer

MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "bin", "visualize_minicif.py")
spec = importlib.util.spec_from_file_location("visualize_minicif", MODULE_PATH)
visualize_minicif = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visualize_minicif)
prompt_from_minicif = visualize_minicif.prompt_from_minicif
summarize = visualize_minicif.summarize


class VisualizeMinicifTest(unittest.TestCase):
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
            },
        ])

        summary = summarize(df)

        self.assertEqual(summary.loc[0, "valid_minicif_rate"], 0.5)
        self.assertEqual(summary.loc[0, "structure_rate"], 0.5)
        self.assertEqual(summary.loc[0, "element_set_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
