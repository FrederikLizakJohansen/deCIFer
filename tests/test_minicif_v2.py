import unittest

import torch
from pymatgen.core import Lattice, Structure

from decifer.minicif_v2 import (
    FORMULA_TOKEN,
    START_TOKEN,
    MinicifV2Config,
    MinicifV2Tokenizer,
    allowed_minicif_v2_next_token_ids,
    canonicalize_structure_v2,
    mask_minicif_v2_logits,
    minicif_v2_to_structure,
    parse_minicif_v2,
)


class MinicifV2Test(unittest.TestCase):
    def setUp(self):
        self.structure = Structure.from_spacegroup(
            "Fm-3m",
            Lattice.cubic(5.64),
            ["Na", "Cl"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )

    def test_canonical_representation_contains_formula_and_wyckoff_letters(self):
        text = canonicalize_structure_v2(
            self.structure,
            MinicifV2Config(decimal_places=3),
        )

        self.assertIn("<mcif2> Na Cl formula Na 1 Cl 1 cs_7 sg_225", text)
        self.assertIn("<atom> Na wp_a", text)
        self.assertIn("<atom> Cl wp_b", text)
        self.assertTrue(text.endswith("</mcif2>"))

    def test_tokenizer_round_trip_and_typed_vocab_partition(self):
        text = canonicalize_structure_v2(self.structure)
        tokenizer = MinicifV2Tokenizer()

        ids = tokenizer.encode(tokenizer.tokenize_minicif(text))
        groups = tokenizer.token_type_ids

        self.assertEqual(tokenizer.decode(ids), text)
        self.assertEqual(sum(len(group) for group in groups.values()), tokenizer.vocab_size)
        self.assertEqual(len(set().union(*(set(group) for group in groups.values()))), tokenizer.vocab_size)

    def test_round_trip_preserves_composition_space_group_and_wyckoff(self):
        text = canonicalize_structure_v2(self.structure, MinicifV2Config(decimal_places=4))

        parsed = parse_minicif_v2(text)
        restored = minicif_v2_to_structure(text)

        self.assertEqual(parsed.formula, {"Na": 1, "Cl": 1})
        self.assertEqual(parsed.space_group, 225)
        self.assertEqual(restored.composition.reduced_formula, "NaCl")
        self.assertEqual(restored.get_space_group_info()[1], 225)

    def test_round_trip_covers_all_crystal_systems(self):
        cases = [
            Structure(
                Lattice.from_parameters(4, 5, 6, 75, 80, 85),
                ["Si", "O"],
                [[0.137, 0.231, 0.319], [0.411, 0.087, 0.733]],
            ),
            Structure.from_spacegroup("P2/m", Lattice.monoclinic(4, 5, 6, 105), ["Si"], [[0.137, 0.231, 0.319]]),
            Structure.from_spacegroup("Pmmm", Lattice.orthorhombic(4, 5, 6), ["Si"], [[0.137, 0.231, 0.319]]),
            Structure.from_spacegroup("P4/mmm", Lattice.tetragonal(4, 6), ["Si"], [[0.137, 0.231, 0.319]]),
            Structure.from_spacegroup("P-3m1", Lattice.hexagonal(4, 6), ["Si"], [[0.137, 0.231, 0.319]]),
            Structure.from_spacegroup("P6/mmm", Lattice.hexagonal(4, 6), ["Si"], [[0.137, 0.231, 0.319]]),
            Structure.from_spacegroup("Pm-3m", Lattice.cubic(4), ["Si"], [[0.137, 0.231, 0.319]]),
        ]

        for expected_crystal_system, structure in enumerate(cases, start=1):
            with self.subTest(crystal_system=expected_crystal_system):
                text = canonicalize_structure_v2(structure)
                parsed = parse_minicif_v2(text)
                restored = minicif_v2_to_structure(text)

                self.assertEqual(parsed.crystal_system, expected_crystal_system)
                self.assertEqual(restored.get_space_group_info()[1], parsed.space_group)
                self.assertEqual(
                    restored.composition.reduced_formula,
                    structure.composition.reduced_formula,
                )

    def test_constituent_only_prefix_can_transition_to_formula(self):
        tokenizer = MinicifV2Tokenizer()
        prefix = "<mcif2> Na Cl "
        ids = tokenizer.encode(tokenizer.tokenize_minicif(prefix))

        allowed = allowed_minicif_v2_next_token_ids(ids, tokenizer)

        self.assertIn(tokenizer.token_to_id[FORMULA_TOKEN], allowed)
        self.assertNotIn(tokenizer.token_to_id["Na"], allowed)

    def test_formula_prefix_can_transition_to_crystal_system(self):
        tokenizer = MinicifV2Tokenizer()
        prefix = "<mcif2> Na Cl formula Na 1 Cl 1 "
        ids = tokenizer.encode(tokenizer.tokenize_minicif(prefix))

        allowed = allowed_minicif_v2_next_token_ids(ids, tokenizer)

        self.assertIn(tokenizer.token_to_id["cs_7"], allowed)
        self.assertNotIn(tokenizer.token_to_id["Fe"], allowed)

    def test_lattice_mask_forces_cubic_b(self):
        tokenizer = MinicifV2Tokenizer()
        prefix = "<mcif2> Na Cl formula Na 1 Cl 1 cs_7 sg_225 cell 5.640 "
        ids = tokenizer.encode(tokenizer.tokenize_minicif(prefix))
        logits = torch.zeros(1, tokenizer.vocab_size)

        masked = mask_minicif_v2_logits(logits, torch.tensor([ids]), tokenizer)

        self.assertEqual(masked[0, tokenizer.token_to_id["5"]].item(), 0)
        self.assertTrue(torch.isneginf(masked[0, tokenizer.token_to_id["6"]]))

    def test_parser_rejects_formula_that_disagrees_with_expanded_structure(self):
        text = canonicalize_structure_v2(self.structure).replace("formula Na 1 Cl 1", "formula Na 2 Cl 1")

        with self.assertRaisesRegex(ValueError, "does not match formula"):
            minicif_v2_to_structure(text)

    def test_parser_rejects_wrong_wyckoff_letter(self):
        text = canonicalize_structure_v2(self.structure).replace("<atom> Na wp_a", "<atom> Na wp_c")

        with self.assertRaisesRegex(ValueError, "does not match wp_c"):
            minicif_v2_to_structure(text)

    def test_parser_rejects_invalid_occupancy(self):
        text = canonicalize_structure_v2(self.structure).replace("1.0000 </mcif2>", "1.1000 </mcif2>")

        with self.assertRaisesRegex(ValueError, "occupancy must be in"):
            parse_minicif_v2(text)

    def test_empty_prompt_starts_with_v2_token(self):
        tokenizer = MinicifV2Tokenizer()

        allowed = allowed_minicif_v2_next_token_ids([], tokenizer)

        self.assertEqual(allowed, {tokenizer.token_to_id[START_TOKEN]})


if __name__ == "__main__":
    unittest.main()
