from decifer.tokenizer import Tokenizer


def test_spacegroup_tokenization_disambiguates_symbol():
    tokenizer = Tokenizer()
    cif = "_symmetry_space_group_name_H-M Pm\n"

    tokens = tokenizer.tokenize_cif(cif)

    assert "Pm_sg" in tokens


def test_encode_decode_roundtrip_for_basic_prompt():
    tokenizer = Tokenizer()
    cif = "data_CeO2\n_chemical_formula_sum 'Ce1 O2'\n"

    tokens = tokenizer.tokenize_cif(cif)
    encoded = tokenizer.encode(tokens)
    decoded = tokenizer.decode(encoded)

    assert decoded.startswith("data_CeO2\n")
    assert "_chemical_formula_sum" in decoded
