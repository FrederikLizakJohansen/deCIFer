import math
import re
import pandas as pd
import numpy as np

from pymatgen.core import Composition
from pymatgen.io.cif import CifBlock
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.core.operations import SymmOp

def get_atomic_props_block(composition, oxi=False):
    noble_vdw_radii = {
        "He": 1.40,
        "Ne": 1.54,
        "Ar": 1.88,
        "Kr": 2.02,
        "Xe": 2.16,
        "Rn": 2.20,
    }

    allen_electronegativity = {
        "He": 4.16,
        "Ne": 4.79,
        "Ar": 3.24,
    }

    def _format(val):
        return f"{float(val): .4f}"

    def _format_X(elem):
        if math.isnan(elem.X) and str(elem) in allen_electronegativity:
            return allen_electronegativity[str(elem)]
        return _format(elem.X)

    def _format_radius(elem):
        if elem.atomic_radius is None and str(elem) in noble_vdw_radii:
            return noble_vdw_radii[str(elem)]
        return _format(elem.atomic_radius)

    props = {str(el): (_format_X(el), _format_radius(el), _format(el.average_ionic_radius))
             for el in sorted(composition.elements)}

    data = {}
    data["_atom_type_symbol"] = list(props)
    data["_atom_type_electronegativity"] = [v[0] for v in props.values()]
    data["_atom_type_radius"] = [v[1] for v in props.values()]
    # use the average ionic radius
    data["_atom_type_ionic_radius"] = [v[2] for v in props.values()]

    loop_vals = [
        "_atom_type_symbol",
        "_atom_type_electronegativity",
        "_atom_type_radius",
        "_atom_type_ionic_radius"
    ]

    if oxi:
        symbol_to_oxinum = {str(el): (float(el.oxi_state), _format(el.ionic_radius)) for el in sorted(composition.elements)}
        data["_atom_type_oxidation_number"] = [v[0] for v in symbol_to_oxinum.values()]
        # if we know the oxidation state of the element, use the ionic radius for the given oxidation state
        data["_atom_type_ionic_radius"] = [v[1] for v in symbol_to_oxinum.values()]
        loop_vals.append("_atom_type_oxidation_number")

    loops = [loop_vals]

    return str(CifBlock(data, loops, "")).replace("data_\n", "")

def extract_species(cif_str):
    return list(set(Composition(extract_formula_nonreduced(cif_str)).as_dict().keys()))

def extract_composition(cif_str):
    return Composition(extract_formula_nonreduced(cif_string)).as_dict()

def add_atomic_props_block(cif_str, oxi=False):
    comp = Composition(extract_formula_nonreduced(cif_str))

    block = get_atomic_props_block(composition=comp, oxi=oxi)

    # the hypothesis is that the atomic properties should be the first thing
    #  that the model must learn to associate with the composition, since
    #  they will determine so much of what follows in the file
    pattern = r"_symmetry_space_group_name_H-M"
    match = re.search(pattern, cif_str)

    if match:
        start_pos = match.start()
        modified_cif = cif_str[:start_pos] + block + "\n" + cif_str[start_pos:]
        return modified_cif
    else:
        raise Exception(f"Pattern not found: {cif_str}")

def replace_symmetry_loop_with_P1(cif_str):
    start = cif_str.find("_symmetry_equiv_pos_site_id")
    end = cif_str.find("loop_", start+1)

    replacement = """_symmetry_equiv_pos_site_id\n_symmetry_equiv_pos_as_xyz\n1  'x, y, z'\n"""

    return cif_str.replace(cif_str[start:end], replacement)

def reinstate_symmetry_loop(cif_str, space_group_symbol):
    space_group = SpaceGroup(space_group_symbol)
    symmetry_ops = space_group.symmetry_ops

    loops = []
    data = {}
    symmops = []
    for op in symmetry_ops:
        v = op.translation_vector
        symmops.append(SymmOp.from_rotation_and_translation(op.rotation_matrix, v))

    try:
        ops = [op.as_xyz_string() for op in symmops]
    except:
        ops = [op.as_xyz_str() for op in symmops]
    data["_symmetry_equiv_pos_site_id"] = [f"{i}" for i in range(1, len(ops) + 1)]
    data["_symmetry_equiv_pos_as_xyz"] = ops

    loops.append(["_symmetry_equiv_pos_site_id", "_symmetry_equiv_pos_as_xyz"])

    symm_block = str(CifBlock(data, loops, "")).replace("data_\n", "")

    pattern = r"(loop_\n_symmetry_equiv_pos_site_id\n_symmetry_equiv_pos_as_xyz\n1  'x, y, z')"
    cif_str_updated = re.sub(pattern, symm_block, cif_str)

    return cif_str_updated

def remove_cif_header(cif_str):
    lines = cif_str.split('\n')
    cif_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 0 and not line.startswith("#") and "pymatgen" not in line:
            cif_lines.append(line)

    cif_str = '\n'.join(cif_lines)
    return cif_str

OXI_LOOP_PATTERN = r'loop_[^l]*(?:l(?!oop_)[^l]*)*_atom_type_oxidation_number[^l]*(?:l(?!oop_)[^l]*)*'
OXI_STATE_PATTERN = r'(\n\s+)([A-Za-z]+)[\d.+-]*'

# Regular expression to match occupancy values
OCCU_PATTERN = re.compile(
        r'(_atom_site_type_symbol[\s\S]+?_atom_site_occupancy\s+)((?:\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\d+\.\d+\s*)+)', re.MULTILINE)

def remove_oxidation_loop(cif_str):
    cif_str = re.sub(OXI_LOOP_PATTERN, '', cif_str)
    cif_str = re.sub(OXI_STATE_PATTERN, r'\1\2', cif_str)
    return cif_str

def format_occupancies(cif_str, decimal_places=4):

    # Search for the occupancy block
    occupancy_block_match = OCCU_PATTERN.search(cif_str)
    
    if occupancy_block_match:
        # Extract header and atoms
        header = occupancy_block_match.group(1)
        atoms = occupancy_block_match.group(2).strip().split('\n')
        
        # Check occupancies and format them
        formatted_atoms = []
        occupancies = []
        for atom in atoms:
            parts = atom.split()
            occupancy = float(parts[-1])
            occupancies.append(occupancy)
            formatted_occupancy = "{:.{}f}".format(occupancy,decimal_places)
            formatted_atoms.append('  '.join(parts[:-1]) + f"  {formatted_occupancy}")

        formatted_block = header + '\n  '.join(formatted_atoms)
        cif_str = OCCU_PATTERN.sub(formatted_block, cif_str)

    return cif_str

def extract_formula_units(cif_str):
    return extract_numeric_property(cif_str, "_cell_formula_units_Z", numeric_type=int)

def extract_space_group_symbol(cif_str):
    match = re.search(r"_symmetry_space_group_name_H-M\s+('([^']+)'|(\S+))", cif_str)
    if match:
        return match.group(2) if match.group(2) else match.group(3)
    raise Exception(f"could not extract space group from:\n{cif_str}")

def extract_numeric_property(cif_str, prop, numeric_type=float):
    match = re.search(rf"{prop}\s+([.0-9]+)", cif_str)
    if match:
        return numeric_type(match.group(1))
    raise Exception(f"could not find {prop} in:\n{cif_str}")

def extract_data_formula(cif_str):
    #match = re.search(r"data_([A-Za-z0-9]+)\n", cif_str)
    match = re.search(r"data_([A-Za-z0-9.()]+)\n", cif_str) # Including paranthesis
    if match:
        return match.group(1)
    raise Exception(f"could not find data_ in:\n{cif_str}")

def extract_formula_nonreduced(cif_str):
    match = re.search(r"_chemical_formula_sum\s+('([^']+)'|(\S+))", cif_str)
    if match:
        return match.group(2) if match.group(2) else match.group(3)
    raise Exception(f"could not extract _chemical_formula_sum value from:\n{cif_str}")

def replace_data_formula_with_nonreduced_formula(cif_str):
    pattern = r"_chemical_formula_sum\s+(.+)\n"
    pattern_2 = r"(data_)(.*?)(\n)"
    match = re.search(pattern, cif_str)
    if match:
        chemical_formula = match.group(1)
        chemical_formula = chemical_formula.replace("'", "").replace(" ", "")

        modified_cif = re.sub(pattern_2, r'\1' + chemical_formula + r'\3', cif_str)

        return modified_cif
    else:
        raise Exception(f"Chemical formula not found {cif_str}")

def round_numbers(cif_str, decimal_places=4):
    # Pattern to match a floating point number in the CIF file
    # It also matches numbers in scientific notation
    # pattern = r"[-+]?\d*\.\d+([eE][-+]?\d+)?"
    pattern = r"[-+]?\d*\.?\d+([eE][-+]?\d+)?" # Including occupancies

    # Function to round the numbers
    def round_number(match):
        number_str = match.group()
        number = float(number_str)
        # Check if number of digits after decimal point is less than 'decimal_places'
        if len(number_str.split('.')[-1]) <= decimal_places:
            return number_str
        rounded = round(number, decimal_places)
        return format(rounded, '.{}f'.format(decimal_places))

    # Replace all occurrences of the pattern using a regex sub operation
    cif_string_rounded = re.sub(pattern, round_number, cif_str)

    return cif_string_rounded

def get_unit_cell_volume(a, b, c, alpha_deg, beta_deg, gamma_deg):
    alpha_rad = math.radians(alpha_deg)
    beta_rad = math.radians(beta_deg)
    gamma_rad = math.radians(gamma_deg)

    volume = (a * b * c * math.sqrt(1 - math.cos(alpha_rad) ** 2 - math.cos(beta_rad) ** 2 - math.cos(gamma_rad) ** 2 +
                                    2 * math.cos(alpha_rad) * math.cos(beta_rad) * math.cos(gamma_rad)))

    return volume

def extract_volume(cif_str):
    return extract_numeric_property(cif_str, "_cell_volume")

