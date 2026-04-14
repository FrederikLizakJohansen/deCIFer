import math
import sys
import types

from conftest import REPO_ROOT, load_module_from_path


def load_evaluation_module(monkeypatch):
    fake_utility_module = types.ModuleType("decifer.utility")
    fake_utility_module.bond_length_reasonableness_score = lambda cif: 1.25
    fake_utility_module.extract_numeric_property = lambda cif, key, numeric_type=float: {
        "_cell_length_a": 1.0,
        "_cell_length_b": 2.0,
        "_cell_length_c": 3.0,
        "_cell_angle_alpha": 90.0,
        "_cell_angle_beta": 90.0,
        "_cell_angle_gamma": 90.0,
    }[key]
    fake_utility_module.extract_space_group_symbol = lambda cif: "P 1"
    fake_utility_module.extract_species = lambda cif: ["Ce", "O"]
    fake_utility_module.extract_volume = lambda cif: 6.0
    fake_utility_module.get_unit_cell_volume = lambda a, b, c, alpha, beta, gamma: a * b * c
    fake_utility_module.is_atom_site_multiplicity_consistent = lambda cif: True
    fake_utility_module.is_formula_consistent = lambda cif: True
    fake_utility_module.is_space_group_consistent = lambda cif: True
    fake_utility_module.space_group_symbol_to_number = lambda symbol: 1

    monkeypatch.setitem(sys.modules, "decifer.utility", fake_utility_module)

    return load_module_from_path(
        REPO_ROOT / "decifer" / "evaluation.py",
        "test_decifer_evaluation_module",
    )


def test_get_cif_statistics_and_summary(monkeypatch):
    module = load_evaluation_module(monkeypatch)

    stats = module.get_cif_statistics("data_test\n")
    assert stats["validity"] == {
        "formula": True,
        "site_multiplicity": True,
        "bond_length": True,
        "spacegroup": True,
    }
    assert stats["cell_params"]["implied_vol"] == 6.0
    assert stats["cell_params"]["gen_vol"] == 6.0
    assert stats["spacegroup"] == "P 1"
    assert stats["species"] == ["Ce", "O"]

    row = {
        "status": ["task", "syntax", "sensible", "statistics", "success"],
        "validity": stats["validity"],
        "cif_string_sample": "data_sample\n",
        "cif_string_gen": "data_gen\n",
        "xrd_clean_sample": {
            "q": [0.0, 1.0],
            "iq": [1.0, 0.0],
            "q_disc": [0.0, 1.0],
            "iq_disc": [0.7, 0.3],
        },
        "xrd_clean_gen": {
            "q": [0.0, 1.0],
            "iq": [1.0, 0.0],
            "q_disc": [0.0, 1.0],
            "iq_disc": [0.6, 0.4],
        },
        "rmsd": 0.5,
        "seq_len_sample": 10,
        "seq_len_gen": 12,
    }

    summary = module.summarize_successful_evaluation_row(row)

    assert summary is not None
    assert summary["validity"] is True
    assert summary["formula_validity"] is True
    assert summary["spacegroup_num_sample"] == 1
    assert summary["spacegroup_num_gen"] == 1
    assert summary["rmsd"] == 0.5
    assert summary["rwp"] == 0.0
    assert math.isfinite(summary["wd"])
