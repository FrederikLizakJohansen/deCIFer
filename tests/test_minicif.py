import unittest

from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from decifer.minicif import (
    ATOM_TOKEN,
    END_TOKEN,
    START_TOKEN,
    MinicifConfig,
    MinicifTokenizer,
    canonicalize_cif,
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


if __name__ == "__main__":
    unittest.main()
