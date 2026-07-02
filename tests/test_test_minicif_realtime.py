import importlib.util
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import h5py
import numpy as np
from pymatgen.core import Lattice, Structure

from decifer.minicif import MinicifTokenizer
from decifer.pxrd import max_q_for_wavelength

MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "bin", "test_minicif_realtime.py")
spec = importlib.util.spec_from_file_location("test_minicif_realtime", MODULE_PATH)
test_minicif_realtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_minicif_realtime)

prompt_from_minicif = test_minicif_realtime.prompt_from_minicif
print_results = test_minicif_realtime.print_results
read_sample = test_minicif_realtime.read_sample
clean_xrd_kwargs = test_minicif_realtime.clean_xrd_kwargs
refine_best_candidate = test_minicif_realtime.refine_best_candidate
save_fit_figure = test_minicif_realtime.save_fit_figure


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

    def test_clean_xrd_kwargs_clamps_qmax_below_singularity(self):
        args = SimpleNamespace(
            qmin=None,
            qmax=None,
            qstep=None,
            clean_fwhm=None,
            eta=None,
            wavelength="CuKa",
        )

        kwargs = clean_xrd_kwargs({"qmax": 10.0, "fwhm_range_min": 0.04, "fwhm_range_max": 0.08}, args)

        self.assertLess(kwargs["qmax"], max_q_for_wavelength(1.5406))

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

    def test_save_fit_figure_writes_file_for_best_candidate(self):
        structure = Structure(
            Lattice.cubic(3.0),
            ["Na"],
            [[0, 0, 0]],
        )
        q_grid = np.linspace(0, 5, 64)
        reference_iq = np.exp(-((q_grid - 2.0) ** 2))
        generated_iq = np.exp(-((q_grid - 2.1) ** 2))
        rows = [{
            "rep": 0,
            "rwp": 0.12,
            "generated_iq": generated_iq,
            "generated_structure": structure,
        }]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "fit.png")
            save_fit_figure(out_path, q_grid, reference_iq, structure, rows, "toy", 1)

            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)

    def test_refine_best_candidate_keeps_valid_candidate(self):
        structure = Structure(
            Lattice.cubic(3.0),
            ["Na"],
            [[0, 0, 0]],
        )
        q_grid = np.linspace(0, 4, 200)
        reference_iq = np.exp(-((q_grid - 2.0) ** 2))
        rows = [{
            "rep": 0,
            "rwp": 0.5,
            "generated_crystal_system": 7,
            "generated_structure": structure,
        }]
        xrd_kwargs = {
            "qmin": 0.0,
            "qmax": 4.0,
            "qstep": 0.02,
            "fwhm_range": (0.08, 0.08),
            "eta_range": (0.5, 0.5),
            "noise_range": None,
            "intensity_scale_range": None,
            "mask_prob": None,
            "final_normalize": True,
        }

        refined = refine_best_candidate(rows, reference_iq, xrd_kwargs, "CuKa", max_nfev=2)

        self.assertIsNotNone(refined)
        self.assertIn("rwp_after", refined)

    def test_refine_best_candidate_rescues_misscaled_true_structure(self):
        structure_to_continuous_xrd = test_minicif_realtime.structure_to_continuous_xrd
        rwp = test_minicif_realtime.rwp
        xrd_kwargs = {
            "qmin": 0.0,
            "qmax": 6.0,
            "qstep": 0.02,
            "fwhm_range": (0.06, 0.06),
            "eta_range": (0.5, 0.5),
            "noise_range": None,
            "intensity_scale_range": None,
            "mask_prob": None,
            "final_normalize": True,
        }
        # Reference is a tetragonal 2-atom motif; the peak intensities (not just spacing)
        # carry structural information, so refinement of a wrong motif cannot fully fit it.
        true_lattice = Lattice.tetragonal(4.0, 6.0)
        true_structure = Structure(true_lattice, ["Fe", "O"], [[0, 0, 0], [0.0, 0.0, 0.32]])
        reference_iq = structure_to_continuous_xrd(true_structure, xrd_kwargs, "CuKa")

        # Candidate 0 is the correct motif but with a cell scaled ~12% too large, so its raw
        # Rwp is poor. Candidate 1 has a wrong atomic arrangement whose raw Rwp happens to be
        # lower. Only the true motif can be refined (via lattice) to near-zero Rwp.
        misscaled_true = Structure(Lattice.tetragonal(4.5, 6.75), ["Fe", "O"], [[0, 0, 0], [0.0, 0.0, 0.32]])
        wrong_structure = Structure(Lattice.tetragonal(4.0, 6.0), ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.1]])
        misscaled_iq = structure_to_continuous_xrd(misscaled_true, xrd_kwargs, "CuKa")
        wrong_iq = structure_to_continuous_xrd(wrong_structure, xrd_kwargs, "CuKa")
        rows = [
            {"rep": 0, "rwp": rwp(reference_iq, misscaled_iq), "generated_crystal_system": 4,
             "generated_structure": misscaled_true},
            {"rep": 1, "rwp": rwp(reference_iq, wrong_iq), "generated_crystal_system": 4,
             "generated_structure": wrong_structure},
        ]
        self.assertGreater(rows[0]["rwp"], rows[1]["rwp"])  # true structure ranks worse on raw Rwp

        refined = refine_best_candidate(rows, reference_iq, xrd_kwargs, "CuKa", max_nfev=40, top_k=2)

        self.assertIsNotNone(refined)
        self.assertEqual(refined["source_rep"], 0)
        self.assertLess(refined["rwp_after"], rows[1]["rwp"])


if __name__ == "__main__":
    unittest.main()
