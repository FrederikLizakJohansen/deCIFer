import unittest

import torch
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from decifer.minicif import (
    ATOM_TOKEN,
    END_TOKEN,
    START_TOKEN,
    MinicifConfig,
    MinicifTokenizer,
    allowed_minicif_next_token_ids,
    canonicalize_cif,
    canonicalize_cif_block,
    lattice_constraint_violations,
    mask_minicif_logits,
    minicif_to_structure,
    parse_minicif,
)


class MinicifTest(unittest.TestCase):
    def test_canonicalize_cif_keeps_minimal_structure_fields(self):
        structure = Structure(
            Lattice.cubic(5.64),
            ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        cif_string = str(CifWriter(structure, symprec=0.1))

        minicif = canonicalize_cif(cif_string, MinicifConfig(decimal_places=3))

        self.assertTrue(minicif.startswith(START_TOKEN + " "))
        self.assertTrue(minicif.endswith(" " + END_TOKEN))
        self.assertIn("<mcif> Na Cl cs_7", minicif)
        self.assertIn("cs_7", minicif)
        self.assertIn("sg_221", minicif)
        self.assertIn("cell 5.640 5.640 5.640 90.000 90.000 90.000", minicif)
        self.assertEqual(minicif.count(ATOM_TOKEN), 2)
        self.assertIn("<atom> Na 1 0.000 0.000 0.000 1.000", minicif)
        self.assertIn("<atom> Cl 1 0.500 0.500 0.500 1.000", minicif)

    def test_element_order_can_be_alphabetical(self):
        structure = Structure(
            Lattice.cubic(5.64),
            ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        cif_string = str(CifWriter(structure, symprec=0.1))

        minicif = canonicalize_cif(
            cif_string,
            MinicifConfig(decimal_places=2, element_order="alphabetical"),
        )

        self.assertIn("<mcif> Cl Na cs_7", minicif)

    def test_minicif_tokenizer_round_trips_canonical_string(self):
        structure = Structure(
            Lattice.cubic(5.64),
            ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        cif_string = str(CifWriter(structure, symprec=0.1))
        minicif = canonicalize_cif(cif_string, MinicifConfig(decimal_places=3))
        tokenizer = MinicifTokenizer()

        tokens = tokenizer.tokenize_minicif(minicif)
        ids = tokenizer.encode(tokens)

        self.assertEqual(tokenizer.decode(ids), minicif)

    def test_canonicalize_cif_block_uses_parsed_block(self):
        structure = Structure(
            Lattice.cubic(5.64),
            ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        block = next(iter(CifWriter(structure, symprec=0.1).cif_file.data.values())).data

        minicif = canonicalize_cif_block(block, MinicifConfig(decimal_places=3))

        self.assertIn("<mcif> Na Cl cs_7 sg_221", minicif)
        self.assertIn("cell 5.640 5.640 5.640 90.000 90.000 90.000", minicif)
        self.assertEqual(minicif.count(ATOM_TOKEN), 2)

    def test_space_group_choices_are_conditioned_on_crystal_system(self):
        tokenizer = MinicifTokenizer()
        prefix = "<mcif> Na Cl cs_7 "
        ids = tokenizer.encode(tokenizer.tokenize_minicif(prefix))

        allowed = allowed_minicif_next_token_ids(ids, tokenizer)

        self.assertIn(tokenizer.token_to_id["sg_195"], allowed)
        self.assertIn(tokenizer.token_to_id["sg_230"], allowed)
        self.assertNotIn(tokenizer.token_to_id["sg_194"], allowed)

    def test_atom_elements_are_conditioned_on_prefix_elements(self):
        tokenizer = MinicifTokenizer()
        prefix = (
            "<mcif> Na Cl cs_7 sg_221 cell "
            "5.640 5.640 5.640 90.000 90.000 90.000 <atom> "
        )
        ids = tokenizer.encode(tokenizer.tokenize_minicif(prefix))

        allowed = allowed_minicif_next_token_ids(ids, tokenizer)

        self.assertIn(tokenizer.token_to_id["Na"], allowed)
        self.assertIn(tokenizer.token_to_id["Cl"], allowed)
        self.assertNotIn(tokenizer.token_to_id["Fe"], allowed)

    def test_cell_constraints_reject_invalid_cubic_cell(self):
        minicif = (
            "<mcif> Na Cl cs_7 sg_221 cell "
            "5.640 5.700 5.640 90.000 90.000 90.000 "
            "<atom> Na 1 0.000 0.000 0.000 1.000 </mcif>"
        )

        with self.assertRaisesRegex(ValueError, "cell violates lattice constraints"):
            parse_minicif(minicif)

    def test_trigonal_validation_accepts_hexagonal_and_rhombohedral_axes(self):
        self.assertEqual(lattice_constraint_violations((5.0, 5.0, 8.0, 90.0, 90.0, 120.0), 5, 148), [])
        self.assertEqual(lattice_constraint_violations((5.0, 5.0, 5.0, 60.0, 60.0, 60.0), 5, 148), [])

    def test_cell_constraints_force_cubic_b_to_match_a(self):
        tokenizer = MinicifTokenizer()
        prefix = "<mcif> Na Cl cs_7 sg_221 cell 5.640 "
        ids = tokenizer.encode(tokenizer.tokenize_minicif(prefix))
        allowed = allowed_minicif_next_token_ids(ids, tokenizer)

        self.assertEqual(allowed, {tokenizer.token_to_id["5"]})

        prefix = "<mcif> Na Cl cs_7 sg_221 cell 5.640 5.640"
        ids = tokenizer.encode(tokenizer.tokenize_minicif(prefix))
        allowed = allowed_minicif_next_token_ids(ids, tokenizer)

        self.assertEqual(allowed, {tokenizer.token_to_id[" "]})

    def test_cell_constraints_force_hexagonal_gamma_to_120(self):
        tokenizer = MinicifTokenizer()
        prefix = "<mcif> Na Cl cs_6 sg_194 cell 3.200 3.200 5.100 90.000 90.000 "
        ids = tokenizer.encode(tokenizer.tokenize_minicif(prefix))
        allowed = allowed_minicif_next_token_ids(ids, tokenizer)

        self.assertEqual(allowed, {tokenizer.token_to_id["1"]})

    def test_minicif_logit_mask_blocks_invalid_atom_elements(self):
        tokenizer = MinicifTokenizer()
        prefix = (
            "<mcif> Na Cl cs_7 sg_221 cell "
            "5.640 5.640 5.640 90.000 90.000 90.000 <atom> "
        )
        ids = tokenizer.encode(tokenizer.tokenize_minicif(prefix))
        logits = torch.zeros(1, tokenizer.vocab_size)

        masked = mask_minicif_logits(logits, torch.tensor([ids]), tokenizer)

        self.assertEqual(masked[0, tokenizer.token_to_id["Na"]].item(), 0)
        self.assertEqual(masked[0, tokenizer.token_to_id["Cl"]].item(), 0)
        self.assertTrue(torch.isneginf(masked[0, tokenizer.token_to_id["Fe"]]))

    def test_minicif_logit_mask_blocks_invalid_cubic_b_value(self):
        tokenizer = MinicifTokenizer()
        prefix = "<mcif> Na Cl cs_7 sg_221 cell 5.640 "
        ids = tokenizer.encode(tokenizer.tokenize_minicif(prefix))
        logits = torch.zeros(1, tokenizer.vocab_size)

        masked = mask_minicif_logits(logits, torch.tensor([ids]), tokenizer)

        self.assertEqual(masked[0, tokenizer.token_to_id["5"]].item(), 0)
        self.assertTrue(torch.isneginf(masked[0, tokenizer.token_to_id["6"]]))

    def test_minicif_can_be_rendered_to_structure(self):
        minicif = (
            "<mcif> Na Cl cs_7 sg_221 cell "
            "5.640 5.640 5.640 90.000 90.000 90.000 "
            "<atom> Na 1 0.000 0.000 0.000 1.000 "
            "<atom> Cl 1 0.500 0.500 0.500 1.000 </mcif>"
        )

        parsed = parse_minicif(minicif)
        structure = minicif_to_structure(minicif)

        self.assertEqual(parsed.space_group, 221)
        self.assertEqual(structure.composition.reduced_formula, "NaCl")
        self.assertEqual(len(structure), 2)


if __name__ == "__main__":
    unittest.main()
