import importlib.util
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

import h5py
import numpy as np

from decifer.minicif import MinicifTokenizer

MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "bin", "test_minicif_realtime.py")
spec = importlib.util.spec_from_file_location("test_minicif_realtime", MODULE_PATH)
test_minicif_realtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_minicif_realtime)

prompt_from_minicif = test_minicif_realtime.prompt_from_minicif
print_results = test_minicif_realtime.print_results
read_sample = test_minicif_realtime.read_sample


class TestMinicifRealtimeScript(unittest.TestCase):
    def test_read_sample_uses_explicit_index(self):
        minicifs = [
            "<mcif> Na Cl cs_7 sg_221 cell 1.000 1.000 1.000 90.000 90.000 90.000 "
            "<atom> Na 1 0.000 0.000 0.000 1.000 </mcif>",
            "<mcif> Li F cs_7 sg_221 cell 2.000 2.000 2.000 90.000 90.000 90.000 "
            "<atom> Li 1 0.000 0.000 0.000 1.000 </mcif>",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, "test.h5")
            with h5py.File(h5_path, "w") as h5:
                str_dtype = h5py.string_dtype(encoding="utf-8")
                int_vlen = h5py.vlen_dtype(np.dtype("int32"))
                float_vlen = h5py.vlen_dtype(np.dtype("float32"))
                h5.create_dataset("cif_name", data=["row0", "row1"], dtype=str_dtype)
                h5.create_dataset("minicif_string", data=minicifs, dtype=str_dtype)
                h5.create_dataset("cif_tokenized", shape=(2,), dtype=int_vlen)
                q = h5.create_dataset("xrd_disc.q", shape=(2,), dtype=float_vlen)
                iq = h5.create_dataset("xrd_disc.iq", shape=(2,), dtype=float_vlen)
                for i in range(2):
                    h5["cif_tokenized"][i] = np.asarray([0, 1], dtype=np.int32)
                    q[i] = np.asarray([1.0], dtype=np.float32)
                    iq[i] = np.asarray([1.0], dtype=np.float32)

            index, name, minicif, q_disc, iq_disc = read_sample(h5_path, 1, seed=7)

        self.assertEqual(index, 1)
        self.assertEqual(name, "row1")
        self.assertEqual(minicif, minicifs[1])
        self.assertEqual(q_disc.tolist(), [1.0])
        self.assertEqual(iq_disc.tolist(), [1.0])

    def test_prompt_from_minicif_supports_known_field_modes(self):
        tokenizer = MinicifTokenizer()
        minicif = (
            "<mcif> Na Cl cs_7 sg_221 cell "
            "1.000 1.000 1.000 90.000 90.000 90.000 "
            "<atom> Na 1 0.000 0.000 0.000 1.000 </mcif>"
        )

        self.assertEqual(tokenizer.decode(prompt_from_minicif(minicif, "pxrd", tokenizer).tolist()), "<mcif>")
        self.assertEqual(tokenizer.decode(prompt_from_minicif(minicif, "formula", tokenizer).tolist()), "<mcif> Na Cl")
        self.assertEqual(tokenizer.decode(prompt_from_minicif(minicif, "formula-cs", tokenizer).tolist()), "<mcif> Na Cl cs_7")
        self.assertEqual(
            tokenizer.decode(prompt_from_minicif(minicif, "formula-cs-sg", tokenizer).tolist()),
            "<mcif> Na Cl cs_7 sg_221",
        )

    def test_print_results_can_show_decoded_minicifs(self):
        rows = [{
            "rep": 0,
            "parse_ok": True,
            "structure_ok": True,
            "structure_match": False,
            "composition_match": True,
            "space_group_match": True,
            "crystal_system_match": True,
            "rwp": 0.123,
            "rmsd": None,
            "error": "",
            "generated_minicif": "<mcif> Na Cl cs_7 sg_221 cell 1.000 1.000 1.000 90.000 90.000 90.000 </mcif>",
        }]

        output = io.StringIO()
        with redirect_stdout(output):
            print_results(rows, print_minicifs=True)

        self.assertIn("<mcif> Na Cl cs_7 sg_221 cell", output.getvalue())


if __name__ == "__main__":
    unittest.main()
