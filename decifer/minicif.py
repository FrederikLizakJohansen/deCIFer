#!/usr/bin/env python3

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

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
