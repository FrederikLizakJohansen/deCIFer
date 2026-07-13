#!/usr/bin/env python3

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup

from decifer.minicif import (
    CRYSTAL_SYSTEM_SPACE_GROUPS,
    _cell_prefix_is_valid,
    _expected_cell_value_text,
    _format_number,
    _is_complete_number,
    validate_lattice_constraints,
)
from decifer.tokenizer import DIGITS, PAD_TOKEN, UNK_TOKEN
from decifer.utility import space_group_to_crystal_system


START_TOKEN = "<mcif2>"
END_TOKEN = "</mcif2>"
ATOM_TOKEN = "<atom>"
FORMULA_TOKEN = "formula"
CELL_TOKEN = "cell"
WYCKOFF_TOKENS = [f"wp_{letter}" for letter in "abcdefghijklmnopqrstuvwxyz"]


try:
    parser_from_string = CifParser.from_str
except AttributeError:
    parser_from_string = CifParser.from_string


@dataclass
class MinicifV2Config:
    decimal_places: int = 4
    element_order: str = "atomic_number"
    symprec: float = 0.1
    angle_tolerance: float = 5.0


@dataclass
class MinicifV2Atom:
    element: str
    wyckoff: str
    x: float
    y: float
    z: float
    occupancy: float


@dataclass
class ParsedMinicifV2:
    elements: List[str]
    formula: Dict[str, int]
    crystal_system: int
    space_group: int
    cell: Tuple[float, float, float, float, float, float]
    atoms: List[MinicifV2Atom]


class MinicifV2Tokenizer:
    def __init__(self):
        self._tokens = [START_TOKEN, END_TOKEN, ATOM_TOKEN, FORMULA_TOKEN, CELL_TOKEN]
        self._tokens.extend(str(Element.from_Z(z)) for z in range(1, 119))
        self._tokens.extend(f"cs_{i}" for i in range(1, 8))
        self._tokens.extend(f"sg_{i}" for i in range(1, 231))
        self._tokens.extend(WYCKOFF_TOKENS)
        self._tokens.extend(DIGITS)
        self._tokens.extend([".", "+", "-", " "])

        escaped = sorted((re.escape(token) for token in self._tokens), key=len, reverse=True)
        self._token_pattern = "|".join(escaped)
        tokens_with_special = self._tokens + [UNK_TOKEN, PAD_TOKEN]
        self._token_to_id = {token: index for index, token in enumerate(tokens_with_special)}
        self._id_to_token = {index: token for token, index in self._token_to_id.items()}
        self.vocab_size = len(tokens_with_special)

    @property
    def padding_id(self):
        return self._token_to_id[PAD_TOKEN]

    @property
    def token_to_id(self):
        return dict(self._token_to_id)

    @property
    def id_to_token(self):
        return dict(self._id_to_token)

    @property
    def token_type_ids(self):
        element_tokens = {str(Element.from_Z(z)) for z in range(1, 119)}
        symmetry_tokens = {
            token for token in self._token_to_id
            if token.startswith("cs_") or token.startswith("sg_") or token.startswith("wp_")
        }
        numeric_tokens = set(DIGITS + [".", "+", "-"])
        groups = {"element": [], "symmetry": [], "numeric": [], "control": []}
        for token, token_id in self._token_to_id.items():
            if token in element_tokens:
                groups["element"].append(token_id)
            elif token in symmetry_tokens:
                groups["symmetry"].append(token_id)
            elif token in numeric_tokens:
                groups["numeric"].append(token_id)
            else:
                groups["control"].append(token_id)
        return groups

    def encode(self, tokens):
        return [self._token_to_id[token] for token in tokens]

    def decode(self, ids):
        return "".join(self._id_to_token[int(token_id)] for token_id in ids)

    def tokenize_minicif(self, minicif_string: str):
        tokens = re.findall(self._token_pattern, minicif_string)
        return [token if token in self._token_to_id else UNK_TOKEN for token in tokens]


def canonicalize_cif_v2(cif_string: str, config: Optional[MinicifV2Config] = None) -> str:
    config = config or MinicifV2Config()
    parser = parser_from_string(cif_string)
    if hasattr(parser, "parse_structures"):
        structure = parser.parse_structures(primitive=False)[0]
    else:
        structure = parser.get_structures(primitive=False)[0]
    return canonicalize_structure_v2(structure, config)


def canonicalize_structure_v2(structure: Structure, config: Optional[MinicifV2Config] = None) -> str:
    config = config or MinicifV2Config()
    analyzer = SpacegroupAnalyzer(
        structure,
        symprec=config.symprec,
        angle_tolerance=config.angle_tolerance,
    )
    refined = analyzer.get_refined_structure()
    refined_analyzer = SpacegroupAnalyzer(
        refined,
        symprec=config.symprec,
        angle_tolerance=config.angle_tolerance,
    )
    symmetrized = refined_analyzer.get_symmetrized_structure()
    space_group = int(symmetrized.spacegroup.int_number)
    crystal_system = space_group_to_crystal_system(space_group)
    cell = (
        refined.lattice.a,
        refined.lattice.b,
        refined.lattice.c,
        refined.lattice.alpha,
        refined.lattice.beta,
        refined.lattice.gamma,
    )
    validate_lattice_constraints(cell, crystal_system, space_group)

    composition, _ = refined.composition.get_reduced_composition_and_factor()
    formula = _integer_formula(composition)
    elements = _ordered_elements(list(formula), config.element_order)
    atoms = []
    for equivalent_sites, wyckoff_symbol in zip(symmetrized.equivalent_sites, symmetrized.wyckoff_symbols):
        representative = equivalent_sites[0]
        letter = _wyckoff_letter(wyckoff_symbol)
        for element, occupancy in representative.species.items():
            atoms.append(MinicifV2Atom(
                element=str(element),
                wyckoff=f"wp_{letter}",
                x=float(representative.frac_coords[0]),
                y=float(representative.frac_coords[1]),
                z=float(representative.frac_coords[2]),
                occupancy=float(occupancy),
            ))
    atoms.sort(key=lambda atom: (
        Element(atom.element).Z,
        atom.wyckoff,
        atom.x,
        atom.y,
        atom.z,
        atom.occupancy,
    ))

    parts = [START_TOKEN, *elements, FORMULA_TOKEN]
    for element in elements:
        parts.extend([element, str(formula[element])])
    parts.extend([
        f"cs_{crystal_system}",
        f"sg_{space_group}",
        CELL_TOKEN,
        *[_format_number(value, config.decimal_places) for value in cell],
    ])
    for atom in atoms:
        parts.extend([
            ATOM_TOKEN,
            atom.element,
            atom.wyckoff,
            _format_number(atom.x, config.decimal_places),
            _format_number(atom.y, config.decimal_places),
            _format_number(atom.z, config.decimal_places),
            _format_number(atom.occupancy, config.decimal_places),
        ])
    parts.append(END_TOKEN)
    return " ".join(parts)


def parse_minicif_v2(minicif_string: str) -> ParsedMinicifV2:
    fields = minicif_string.strip().split()
    if not fields or fields[0] != START_TOKEN:
        raise ValueError("minicif_v2 does not start with <mcif2>")

    index = 1
    elements = []
    while index < len(fields) and fields[index] != FORMULA_TOKEN:
        _require_element(fields[index])
        elements.append(fields[index])
        index += 1
    if not elements or index >= len(fields):
        raise ValueError("minicif_v2 is missing constituents or formula")
    if len(set(elements)) != len(elements):
        raise ValueError("minicif_v2 constituent elements must be unique")
    index += 1

    formula = {}
    while index < len(fields) and not fields[index].startswith("cs_"):
        if index + 1 >= len(fields):
            raise ValueError("incomplete formula entry")
        element = fields[index]
        if element not in elements or element in formula:
            raise ValueError(f"invalid formula element: {element}")
        count_text = fields[index + 1]
        if not re.fullmatch(r"[1-9]\d*", count_text):
            raise ValueError(f"formula count must be a positive integer: {count_text}")
        formula[element] = int(count_text)
        index += 2
    if set(formula) != set(elements):
        raise ValueError("formula must contain each constituent element exactly once")

    crystal_system = _require_prefixed_int(_field(fields, index, "crystal system"), "cs_")
    if crystal_system not in CRYSTAL_SYSTEM_SPACE_GROUPS:
        raise ValueError(f"invalid crystal system: {crystal_system}")
    index += 1
    space_group = _require_prefixed_int(_field(fields, index, "space group"), "sg_")
    if space_group not in CRYSTAL_SYSTEM_SPACE_GROUPS[crystal_system]:
        raise ValueError(f"space group {space_group} is incompatible with crystal system {crystal_system}")
    index += 1

    if _field(fields, index, "cell") != CELL_TOKEN:
        raise ValueError("minicif_v2 missing cell token")
    index += 1
    if index + 6 > len(fields):
        raise ValueError("minicif_v2 missing cell parameters")
    cell = tuple(float(value) for value in fields[index:index + 6])
    validate_lattice_constraints(cell, crystal_system, space_group)
    index += 6

    atoms = []
    while index < len(fields):
        if fields[index] == END_TOKEN:
            if index != len(fields) - 1:
                raise ValueError("tokens found after </mcif2>")
            if not atoms:
                raise ValueError("minicif_v2 does not contain atom rows")
            return ParsedMinicifV2(elements, formula, crystal_system, space_group, cell, atoms)
        if fields[index] != ATOM_TOKEN or index + 7 > len(fields):
            raise ValueError("incomplete minicif_v2 atom row")
        element = fields[index + 1]
        if element not in elements:
            raise ValueError(f"atom element {element} is not in minicif_v2 constituents")
        wyckoff = fields[index + 2]
        if wyckoff not in WYCKOFF_TOKENS:
            raise ValueError(f"invalid Wyckoff token: {wyckoff}")
        occupancy = float(fields[index + 6])
        if not np.isfinite(occupancy) or not 0 < occupancy <= 1:
            raise ValueError(f"occupancy must be in (0, 1], found {fields[index + 6]}")
        atoms.append(MinicifV2Atom(
            element=element,
            wyckoff=wyckoff,
            x=float(fields[index + 3]),
            y=float(fields[index + 4]),
            z=float(fields[index + 5]),
            occupancy=occupancy,
        ))
        index += 7
    raise ValueError("minicif_v2 missing </mcif2>")


def minicif_v2_to_structure(minicif_string: str, symprec: float = 0.02) -> Structure:
    parsed = parse_minicif_v2(minicif_string)
    lattice = Lattice.from_parameters(*parsed.cell)
    species = [atom.element if atom.occupancy >= 1.0 else {atom.element: atom.occupancy} for atom in parsed.atoms]
    coords = [[atom.x, atom.y, atom.z] for atom in parsed.atoms]
    symbol = SpaceGroup.from_int_number(parsed.space_group).symbol
    structure = Structure.from_spacegroup(symbol, lattice, species, coords)
    _validate_formula(parsed.formula, structure.composition)
    _validate_wyckoff_atoms(parsed.atoms, structure, parsed.space_group, symprec)
    return structure


def allowed_minicif_v2_next_token_ids(
    token_ids,
    tokenizer: Optional[MinicifV2Tokenizer] = None,
) -> Optional[Set[int]]:
    tokenizer = tokenizer or MinicifV2Tokenizer()
    token_to_id = tokenizer.token_to_id
    ids = [int(token_id) for token_id in token_ids if int(token_id) != tokenizer.padding_id]
    text = tokenizer.decode(ids) if ids else ""
    if not text:
        return {token_to_id[START_TOKEN]}

    fields = text.split(" ")
    fresh_field = text.endswith(" ")
    completed = fields[:-1] if fresh_field else fields
    if not fresh_field:
        current = completed[-1]
        expected = _expected_next_field(completed[:-1])
        if expected is not None and expected["kind"] in {"number", "formula_count"}:
            return _numeric_next_ids(current, token_to_id, expected)
        if current == END_TOKEN:
            return {tokenizer.padding_id}
        return {token_to_id[" "]}

    expected = _expected_next_field(completed)
    if expected is None:
        return None
    kind = expected["kind"]
    if kind == "constituent_or_formula":
        allowed = _element_ids(tokenizer, exclude=expected.get("elements", []))
        if expected.get("elements"):
            allowed.add(token_to_id[FORMULA_TOKEN])
        return allowed
    if kind == "formula_element_or_cs":
        remaining = set(expected["elements"]) - set(expected["formula"])
        allowed = {token_to_id[element] for element in remaining}
        if not remaining and expected["formula"]:
            allowed.update(token_to_id[f"cs_{value}"] for value in CRYSTAL_SYSTEM_SPACE_GROUPS)
        return allowed
    if kind == "space_group":
        return {token_to_id[f"sg_{value}"] for value in CRYSTAL_SYSTEM_SPACE_GROUPS[expected["crystal_system"]]}
    if kind == "cell":
        return {token_to_id[CELL_TOKEN]}
    if kind == "number":
        exact = expected.get("exact_text")
        if exact is not None:
            return {token_to_id[exact[0]]}
        return {token_to_id[token] for token in DIGITS + [".", "+", "-"]}
    if kind == "formula_count":
        return {token_to_id[token] for token in DIGITS if token != "0"}
    if kind == "atom_or_end":
        return {token_to_id[ATOM_TOKEN], token_to_id[END_TOKEN]}
    if kind == "atom_element":
        return {token_to_id[element] for element in expected["elements"]}
    if kind == "wyckoff":
        return {token_to_id[token] for token in WYCKOFF_TOKENS}
    if kind == "pad":
        return {tokenizer.padding_id}
    return None


def mask_minicif_v2_logits(logits, sequences, tokenizer: Optional[MinicifV2Tokenizer] = None):
    import torch

    tokenizer = tokenizer or MinicifV2Tokenizer()
    masked = logits.clone()
    for row_index in range(sequences.size(0)):
        allowed = allowed_minicif_v2_next_token_ids(sequences[row_index].detach().cpu().tolist(), tokenizer)
        if not allowed:
            continue
        row_mask = torch.ones(masked.size(-1), dtype=torch.bool, device=masked.device)
        row_mask[list(allowed)] = False
        masked[row_index, row_mask] = -float("inf")
    return masked


def _expected_next_field(fields: List[str]) -> Optional[Dict]:
    if not fields:
        return {"kind": "start"}
    if fields[0] != START_TOKEN:
        return None
    index = 1
    elements = []
    while index < len(fields) and fields[index] != FORMULA_TOKEN:
        if fields[index] not in _element_symbols() or fields[index] in elements:
            return None
        elements.append(fields[index])
        index += 1
    if index == len(fields):
        return {"kind": "constituent_or_formula", "elements": elements}
    if not elements:
        return None
    index += 1

    formula = {}
    while index < len(fields) and not fields[index].startswith("cs_"):
        element = fields[index]
        if element not in elements or element in formula:
            return None
        index += 1
        if index == len(fields):
            return {"kind": "formula_count"}
        if not re.fullmatch(r"[1-9]\d*", fields[index]):
            return None
        formula[element] = int(fields[index])
        index += 1
    if index == len(fields):
        return {"kind": "formula_element_or_cs", "elements": elements, "formula": formula}
    if set(formula) != set(elements):
        return None

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

    cell_values = []
    for cell_index in range(6):
        if index == len(fields):
            return {
                "kind": "number",
                "cell_index": cell_index,
                "exact_text": _expected_cell_value_text(
                    crystal_system, space_group, cell_values, cell_index
                ),
            }
        cell_values.append(fields[index])
        if not _cell_prefix_is_valid(crystal_system, space_group, cell_values):
            return None
        index += 1

    while index < len(fields):
        if fields[index] == END_TOKEN:
            return {"kind": "pad"} if index == len(fields) - 1 else None
        if fields[index] != ATOM_TOKEN:
            return None
        index += 1
        if index == len(fields):
            return {"kind": "atom_element", "elements": elements}
        if fields[index] not in elements:
            return None
        index += 1
        if index == len(fields):
            return {"kind": "wyckoff"}
        if fields[index] not in WYCKOFF_TOKENS:
            return None
        index += 1
        for _ in range(4):
            if index == len(fields):
                return {"kind": "number"}
            if not _is_complete_number(fields[index]):
                return None
            index += 1
    return {"kind": "atom_or_end"}


def _numeric_next_ids(current: str, token_to_id: Dict[str, int], expected: Dict) -> Set[int]:
    exact = expected.get("exact_text")
    if exact is not None:
        if not exact.startswith(current):
            return set()
        if current == exact:
            return {token_to_id[" "]}
        return {token_to_id[exact[len(current)]]}
    allowed = {token_to_id[digit] for digit in DIGITS}
    if expected["kind"] == "formula_count":
        if not current:
            allowed.discard(token_to_id["0"])
        if re.fullmatch(r"[1-9]\d*", current):
            allowed.add(token_to_id[" "])
        return allowed
    if "." not in current:
        allowed.add(token_to_id["."])
    if not current:
        allowed.update([token_to_id["+"], token_to_id["-"]])
    if _is_complete_number(current):
        allowed.add(token_to_id[" "])
    return allowed


def _validate_formula(formula: Dict[str, int], composition: Composition) -> None:
    reduced, _ = composition.get_reduced_composition_and_factor()
    actual = _integer_formula(reduced)
    if actual != formula:
        raise ValueError(f"generated composition {actual} does not match formula {formula}")


def _validate_wyckoff_atoms(
    atoms: List[MinicifV2Atom],
    structure: Structure,
    expected_space_group: int,
    symprec: float,
) -> None:
    analyzer = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=5.0)
    if analyzer.get_space_group_number() != expected_space_group:
        raise ValueError("generated coordinates lower the declared space-group symmetry")
    symmetrized = analyzer.get_symmetrized_structure()
    available = []
    for sites, symbol in zip(symmetrized.equivalent_sites, symmetrized.wyckoff_symbols):
        available.append((sites, f"wp_{_wyckoff_letter(symbol)}"))
    for atom in atoms:
        matched = False
        for sites, wyckoff in available:
            if wyckoff != atom.wyckoff:
                continue
            for site in sites:
                if atom.element not in {str(element) for element in site.species.elements}:
                    continue
                delta = np.asarray(site.frac_coords) - np.asarray([atom.x, atom.y, atom.z])
                delta -= np.round(delta)
                if np.max(np.abs(delta)) <= max(symprec, 10 ** -3):
                    matched = True
                    break
            if matched:
                break
        if not matched:
            raise ValueError(
                f"atom {atom.element} at {(atom.x, atom.y, atom.z)} does not match {atom.wyckoff}"
            )


def _integer_formula(composition: Composition) -> Dict[str, int]:
    formula = {}
    for element, amount in composition.items():
        rounded = int(round(float(amount)))
        if rounded <= 0 or not np.isclose(float(amount), rounded, atol=1e-6):
            raise ValueError(f"minicif_v2 requires integral reduced stoichiometry, found {composition}")
        formula[str(element)] = rounded
    return formula


def _ordered_elements(elements: List[str], order: str) -> List[str]:
    if order == "as_seen":
        return elements
    if order == "alphabetical":
        return sorted(elements)
    if order == "atomic_number":
        return sorted(elements, key=lambda symbol: Element(symbol).Z)
    raise ValueError(f"unknown element_order: {order}")


def _wyckoff_letter(symbol: str) -> str:
    match = re.search(r"([a-z])$", str(symbol).lower())
    if not match:
        raise ValueError(f"unsupported Wyckoff symbol: {symbol}")
    return match.group(1)


def _element_symbols() -> Set[str]:
    return {str(Element.from_Z(z)) for z in range(1, 119)}


def _element_ids(tokenizer: MinicifV2Tokenizer, exclude=()) -> Set[int]:
    excluded = set(exclude)
    return {
        tokenizer.token_to_id[symbol]
        for symbol in _element_symbols()
        if symbol not in excluded
    }


def _require_element(value: str) -> None:
    if value not in _element_symbols():
        raise ValueError(f"invalid element: {value}")


def _field(fields: List[str], index: int, name: str) -> str:
    if index >= len(fields):
        raise ValueError(f"minicif_v2 missing {name}")
    return fields[index]


def _parse_prefixed_int(value: str, prefix: str) -> Optional[int]:
    if not value.startswith(prefix):
        return None
    try:
        return int(value[len(prefix):])
    except ValueError:
        return None


def _require_prefixed_int(value: str, prefix: str) -> int:
    parsed = _parse_prefixed_int(value, prefix)
    if parsed is None:
        raise ValueError(f"expected {prefix}<integer>, found {value}")
    return parsed
