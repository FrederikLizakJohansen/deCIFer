import sys
import types

from conftest import REPO_ROOT, load_module_from_path
from decifer.tokenizer import Tokenizer


def load_generation_module(monkeypatch):
    fake_model_module = types.ModuleType("decifer.decifer_model")
    fake_model_module.Decifer = object
    fake_model_module.DeciferConfig = object
    fake_utility_module = types.ModuleType("decifer.utility")
    fake_utility_module.extract_space_group_symbol = lambda cif: "P 1"
    fake_utility_module.reinstate_symmetry_loop = lambda cif, sg: cif
    fake_utility_module.replace_symmetry_loop_with_P1 = lambda cif: cif
    monkeypatch.setitem(sys.modules, "decifer.decifer_model", fake_model_module)
    monkeypatch.setitem(sys.modules, "decifer.utility", fake_utility_module)
    return load_module_from_path(
        REPO_ROOT / "decifer" / "generation.py",
        "test_decifer_generation_module",
    )


def test_extract_prompt_includes_header_composition_and_spacegroup(monkeypatch):
    module = load_generation_module(monkeypatch)
    tokenizer = Tokenizer()
    cif = (
        "data_demo\n"
        "_chemical_formula_sum Ce1 O2\n"
        "_symmetry_space_group_name_H-M Pm-3m\n"
        "loop_\n"
    )
    sequence = tokenizer.encode(tokenizer.tokenize_cif(cif))

    prompt = module.extract_prompt(sequence, "cpu", add_composition=True, add_spacegroup=True)
    prompt_string = tokenizer.decode(prompt.tolist())

    assert prompt_string.startswith("data_")
    assert "\n_chemical_formula_sum" in prompt_string
    assert "\n_symmetry_space_group_name_H-M" in prompt_string
    assert "loop_" not in prompt_string


def test_extract_prompt_can_include_spacegroup_without_composition(monkeypatch):
    module = load_generation_module(monkeypatch)
    tokenizer = Tokenizer()
    cif = (
        "data_demo\n"
        "_chemical_formula_sum Ce1 O2\n"
        "_symmetry_space_group_name_H-M Pm-3m\n"
        "loop_\n"
    )
    sequence = tokenizer.encode(tokenizer.tokenize_cif(cif))

    prompt = module.extract_prompt(sequence, "cpu", add_composition=False, add_spacegroup=True)
    prompt_string = tokenizer.decode(prompt.tolist())

    assert prompt_string.startswith("data_")
    assert "_symmetry_space_group_name_H-M" in prompt_string
