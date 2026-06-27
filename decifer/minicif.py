#!/usr/bin/env python3

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from pymatgen.core import Element
from pymatgen.io.cif import CifParser

from decifer.tokenizer import DIGITS, PAD_TOKEN, UNK_TOKEN
from decifer.utility import (
    space_group_symbol_to_number,
    space_group_to_crystal_system,
)

try:
    parser_from_string = CifParser.from_str
except AttributeError:
    parser_from_string = CifParser.from_string

START_TOKEN = "<mcif>"
ATOM_TOKEN = "<atom>"
END_TOKEN = "</mcif>"
CELL_TOKEN = "cell"

CRYSTAL_SYSTEM_SPACE_GROUPS = {
    1: range(1, 3),
    2: range(3, 16),
    3: range(16, 75),
    4: range(75, 143),
    5: range(143, 168),
    6: range(168, 195),
    7: range(195, 231),
}


@dataclass
class MinicifConfig:
    decimal_places: int = 4
    element_order: str = "atomic_number"


def canonicalize_cif(cif_string: str, config: Optional[MinicifConfig] = None) -> str:
    """Convert a CIF string into the compact minicif representation."""
    config = config or MinicifConfig()
    parser = parser_from_string(cif_string)
    cif_dict = parser.as_dict()
    block = cif_dict[next(iter(cif_dict))]
    if hasattr(parser, "parse_structures"):
        structure = parser.parse_structures(primitive=True)[0]
    else:
        structure = parser.get_structures()[0]

    elements = _ordered_elements(block, config.element_order)
    space_group_number = _space_group_number(block, structure)
    crystal_system = space_group_to_crystal_system(space_group_number)
    sites = _atom_sites(block)

    parts = [
        START_TOKEN,
        *elements,
        "cs_" + str(crystal_system),
        "sg_" + str(space_group_number),
        CELL_TOKEN,
        _format_number(structure.lattice.a, config.decimal_places),
        _format_number(structure.lattice.b, config.decimal_places),
        _format_number(structure.lattice.c, config.decimal_places),
        _format_number(structure.lattice.alpha, config.decimal_places),
        _format_number(structure.lattice.beta, config.decimal_places),
        _format_number(structure.lattice.gamma, config.decimal_places),
    ]

    for site in sites:
        parts.extend([
            ATOM_TOKEN,
            site["element"],
            str(site["multiplicity"]),
            _format_number(site["x"], config.decimal_places),
            _format_number(site["y"], config.decimal_places),
            _format_number(site["z"], config.decimal_places),
            _format_number(site["occupancy"], config.decimal_places),
        ])

    parts.append(END_TOKEN)
    return " ".join(parts)


class MinicifTokenizer:
    def __init__(self):
        self._tokens = [
            START_TOKEN,
            END_TOKEN,
            ATOM_TOKEN,
            CELL_TOKEN,
        ]
        self._tokens.extend(str(Element.from_Z(z)) for z in range(1, 119))
        self._tokens.extend(f"cs_{i}" for i in range(1, 8))
        self._tokens.extend(f"sg_{i}" for i in range(1, 231))
        self._tokens.extend(DIGITS)
        self._tokens.extend([".", "+", "-", " "])

        self._escaped_tokens = [re.escape(token) for token in self._tokens]
        self._escaped_tokens.sort(key=len, reverse=True)

        self._tokens_with_unk_pad = list(self._tokens)
        self._tokens_with_unk_pad.append(UNK_TOKEN)
        self._tokens_with_unk_pad.append(PAD_TOKEN)

        self._token_to_id = {ch: i for i, ch in enumerate(self._tokens_with_unk_pad)}
        self._id_to_token = {i: ch for i, ch in enumerate(self._tokens_with_unk_pad)}
        self.vocab_size = len(self._tokens_with_unk_pad)

    @property
    def padding_id(self):
        return self._token_to_id[PAD_TOKEN]

    @property
    def token_to_id(self):
        return dict(self._token_to_id)

    @property
    def id_to_token(self):
        return dict(self._id_to_token)

    def encode(self, tokens):
        return [self._token_to_id[t] for t in tokens]

    def decode(self, ids):
        return "".join([self._id_to_token[i] for i in ids])

    def tokenize_minicif(self, minicif_string: str):
        token_pattern = "|".join(self._escaped_tokens)
        tokens = re.findall(token_pattern, minicif_string)
        return [token if token in self._tokens else UNK_TOKEN for token in tokens]


def allowed_minicif_next_token_ids(token_ids, tokenizer: Optional[MinicifTokenizer] = None) -> Optional[Set[int]]:
    """Return allowed next token ids for the minicif DSL, or None if unconstrained."""
    tokenizer = tokenizer or MinicifTokenizer()
    token_to_id = tokenizer.token_to_id
    padding_id = tokenizer.padding_id
    ids = [int(token_id) for token_id in token_ids if int(token_id) != padding_id]
    text = tokenizer.decode(ids) if ids else ""

    if text == "":
        return {token_to_id[START_TOKEN]}

    fields = text.split(" ")
    fresh_field = text.endswith(" ")
    completed_fields = fields[:-1] if fresh_field else fields

    if not fresh_field:
        current = completed_fields[-1]
        if _current_field_is_numeric(completed_fields):
            return _numeric_next_ids(current, token_to_id)
        if current == END_TOKEN:
            return {padding_id}
        return {token_to_id[" "]}

    expected = _expected_next_field(completed_fields)
    if expected is None:
        return None

    kind = expected["kind"]
    if kind == "start":
        return {token_to_id[START_TOKEN]}
    if kind == "formula_element_or_cs":
        allowed = _element_ids(tokenizer)
        if expected["elements"]:
            allowed.update(_crystal_system_ids(tokenizer))
        return allowed
    if kind == "space_group":
        crystal_system = expected["crystal_system"]
        return {token_to_id[f"sg_{i}"] for i in CRYSTAL_SYSTEM_SPACE_GROUPS[crystal_system]}
    if kind == "cell":
        return {token_to_id[CELL_TOKEN]}
    if kind == "number":
        return {token_to_id[token] for token in DIGITS + ["+", "-", "."]}
    if kind == "atom_or_end":
        return {token_to_id[ATOM_TOKEN], token_to_id[END_TOKEN]}
    if kind == "atom_element":
        elements = expected["elements"]
        if not elements:
            return _element_ids(tokenizer)
        return {token_to_id[element] for element in elements if element in token_to_id}
    if kind == "pad":
        return {padding_id}
    return None


def mask_minicif_logits(logits, sequences, tokenizer: Optional[MinicifTokenizer] = None):
    """Set logits for impossible minicif next tokens to -inf."""
    import torch

    tokenizer = tokenizer or MinicifTokenizer()
    masked_logits = logits.clone()
    for row_idx in range(sequences.size(0)):
        allowed = allowed_minicif_next_token_ids(sequences[row_idx].detach().cpu().tolist(), tokenizer)
        if allowed is None:
            continue
        allowed = [token_id for token_id in allowed if 0 <= token_id < masked_logits.size(-1)]
        if not allowed:
            continue
        row_mask = torch.ones(masked_logits.size(-1), dtype=torch.bool, device=masked_logits.device)
        row_mask[allowed] = False
        masked_logits[row_idx, row_mask] = -float("inf")
    return masked_logits


def _ordered_elements(block: Dict, element_order: str) -> List[str]:
    if "_atom_site_type_symbol" in block:
        elements = _unique_preserve_order(_as_list(block["_atom_site_type_symbol"]))
    else:
        raise ValueError("CIF block does not contain _atom_site_type_symbol")

    if element_order == "as_seen":
        return elements
    if element_order == "alphabetical":
        return sorted(elements)
    if element_order == "atomic_number":
        return sorted(elements, key=lambda symbol: Element(symbol).Z)
    raise ValueError(f"unknown element_order: {element_order}")


def _space_group_number(block: Dict, structure) -> int:
    if "_symmetry_Int_Tables_number" in block:
        return int(_parse_number(block["_symmetry_Int_Tables_number"]))

    if "_space_group_IT_number" in block:
        return int(_parse_number(block["_space_group_IT_number"]))

    for key in ["_symmetry_space_group_name_H-M", "_space_group_name_H-M_alt"]:
        if key in block:
            symbol = str(block[key]).strip("'\"")
            number = space_group_symbol_to_number(symbol)
            if number is not None:
                return number

    return int(structure.get_space_group_info()[1])


def _atom_sites(block: Dict) -> List[Dict]:
    required_keys = [
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]
    missing = [key for key in required_keys if key not in block]
    if missing:
        raise ValueError("CIF block missing required atom site keys: " + ", ".join(missing))

    elements = _as_list(block["_atom_site_type_symbol"])
    xs = _as_list(block["_atom_site_fract_x"])
    ys = _as_list(block["_atom_site_fract_y"])
    zs = _as_list(block["_atom_site_fract_z"])
    multiplicities = _as_list(block.get("_atom_site_symmetry_multiplicity", [1] * len(elements)))
    occupancies = _as_list(block.get("_atom_site_occupancy", [1.0] * len(elements)))

    sites = []
    for element, multiplicity, x, y, z, occupancy in zip(elements, multiplicities, xs, ys, zs, occupancies):
        sites.append({
            "element": str(element),
            "multiplicity": int(_parse_number(multiplicity)),
            "x": _parse_number(x),
            "y": _parse_number(y),
            "z": _parse_number(z),
            "occupancy": _parse_number(occupancy),
        })

    return sorted(
        sites,
        key=lambda site: (
            Element(site["element"]).Z,
            site["multiplicity"],
            site["x"],
            site["y"],
            site["z"],
            site["occupancy"],
        ),
    )


def _as_list(value) -> List:
    if isinstance(value, list):
        return value
    return [value]


def _unique_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _parse_number(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.match(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value).strip())
    if not match:
        raise ValueError(f"could not parse numeric value: {value}")
    return float(match.group(0))


def _format_number(value: float, decimal_places: int) -> str:
    value = round(float(value), decimal_places)
    if value == 0:
        value = 0.0
    return f"{value:.{decimal_places}f}"


def _expected_next_field(fields: List[str]) -> Optional[Dict]:
    if not fields or fields == [""]:
        return {"kind": "start"}
    if fields[0] != START_TOKEN:
        return None

    elements = []
    index = 1
    while index < len(fields) and not fields[index].startswith("cs_"):
        if fields[index]:
            elements.append(fields[index])
        index += 1

    if index == len(fields):
        return {"kind": "formula_element_or_cs", "elements": elements}

    crystal_system = _parse_prefixed_int(fields[index], "cs_")
    if crystal_system not in CRYSTAL_SYSTEM_SPACE_GROUPS:
        return None
    index += 1

    if index == len(fields):
        return {"kind": "space_group", "crystal_system": crystal_system}

    space_group = _parse_prefixed_int(fields[index], "sg_")
    if space_group not in CRYSTAL_SYSTEM_SPACE_GROUPS[crystal_system]:
        return None
    index += 1

    if index == len(fields):
        return {"kind": "cell"}
    if fields[index] != CELL_TOKEN:
        return None
    index += 1

    for _ in range(6):
        if index == len(fields):
            return {"kind": "number"}
        index += 1

    while index < len(fields):
        if fields[index] == END_TOKEN:
            return {"kind": "pad"} if index == len(fields) - 1 else None
        if fields[index] != ATOM_TOKEN:
            return None
        index += 1

        if index == len(fields):
            return {"kind": "atom_element", "elements": elements}
        index += 1

        for _ in range(5):
            if index == len(fields):
                return {"kind": "number"}
            index += 1

    return {"kind": "atom_or_end"}


def _current_field_is_numeric(fields: List[str]) -> bool:
    previous_fields = fields[:-1]
    expected = _expected_next_field(previous_fields)
    return expected is not None and expected["kind"] == "number"


def _numeric_next_ids(current: str, token_to_id: Dict[str, int]) -> Set[int]:
    allowed = {token_to_id[digit] for digit in DIGITS}
    if "." not in current:
        allowed.add(token_to_id["."])
    if _is_complete_number(current):
        allowed.add(token_to_id[" "])
    return allowed


def _is_complete_number(value: str) -> bool:
    return re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value) is not None


def _parse_prefixed_int(value: str, prefix: str) -> Optional[int]:
    if not value.startswith(prefix):
        return None
    try:
        return int(value[len(prefix):])
    except ValueError:
        return None


def _element_ids(tokenizer: MinicifTokenizer) -> Set[int]:
    token_to_id = tokenizer.token_to_id
    return {token_to_id[str(Element.from_Z(z))] for z in range(1, 119)}


def _crystal_system_ids(tokenizer: MinicifTokenizer) -> Set[int]:
    token_to_id = tokenizer.token_to_id
    return {token_to_id[f"cs_{i}"] for i in range(1, 8)}
