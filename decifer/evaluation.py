#!/usr/bin/env python3

from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import wasserstein_distance

from decifer.utility import (
    bond_length_reasonableness_score,
    extract_numeric_property,
    extract_space_group_symbol,
    extract_species,
    extract_volume,
    get_unit_cell_volume,
    is_atom_site_multiplicity_consistent,
    is_formula_consistent,
    is_space_group_consistent,
    space_group_symbol_to_number,
)


def safe_extract_boolean(extract_function, *args):
    try:
        return extract_function(*args)
    except Exception:
        return False


def safe_extract(extract_function, *args):
    try:
        return extract_function(*args)
    except Exception:
        return None


def get_cif_statistics(cif_string: str, evaluation_result_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    stat_dict = {
        "validity": {
            "formula": False,
            "site_multiplicity": False,
            "bond_length": False,
            "spacegroup": False,
        },
        "cell_params": {
            "a": None,
            "b": None,
            "c": None,
            "alpha": None,
            "beta": None,
            "gamma": None,
            "implied_vol": None,
            "gen_vol": None,
        },
        "spacegroup": None,
        "species": None,
    }

    stat_dict["validity"]["formula"] = safe_extract_boolean(is_formula_consistent, cif_string)
    stat_dict["validity"]["site_multiplicity"] = safe_extract_boolean(
        is_atom_site_multiplicity_consistent, cif_string
    )
    stat_dict["validity"]["bond_length"] = safe_extract_boolean(
        lambda cif: bond_length_reasonableness_score(cif) >= 1.0, cif_string
    )
    stat_dict["validity"]["spacegroup"] = safe_extract_boolean(is_space_group_consistent, cif_string)

    a = safe_extract(extract_numeric_property, cif_string, "_cell_length_a")
    b = safe_extract(extract_numeric_property, cif_string, "_cell_length_b")
    c = safe_extract(extract_numeric_property, cif_string, "_cell_length_c")
    alpha = safe_extract(extract_numeric_property, cif_string, "_cell_angle_alpha")
    beta = safe_extract(extract_numeric_property, cif_string, "_cell_angle_beta")
    gamma = safe_extract(extract_numeric_property, cif_string, "_cell_angle_gamma")

    implied_vol = safe_extract(get_unit_cell_volume, a, b, c, alpha, beta, gamma)
    gen_vol = safe_extract(extract_volume, cif_string)

    stat_dict["cell_params"].update(
        {
            "a": a,
            "b": b,
            "c": c,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "implied_vol": implied_vol,
            "gen_vol": gen_vol,
        }
    )
    stat_dict["spacegroup"] = safe_extract(extract_space_group_symbol, cif_string)
    stat_dict["species"] = safe_extract(extract_species, cif_string)

    if evaluation_result_dict is not None:
        stat_dict.update(evaluation_result_dict)

    return stat_dict


def get_validity_flags(cif_string: str) -> Dict[str, bool]:
    return get_cif_statistics(cif_string)["validity"]


def is_valid_cif(cif_string: str) -> bool:
    return all(get_validity_flags(cif_string).values())


def residual_weighted_profile(sample, gen):
    sample = np.asarray(sample)
    gen = np.asarray(gen)
    return np.sqrt(np.sum(np.square(sample - gen), axis=-1) / np.sum(np.square(sample), axis=-1))


def spacegroup_info(cif_string: str):
    symbol = extract_space_group_symbol(cif_string)
    number = space_group_symbol_to_number(symbol)
    number = int(number) if number is not None else np.nan
    return symbol, number


def summarize_successful_evaluation_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "success" not in row["status"]:
        return None

    formula_validity = row["validity"]["formula"]
    bond_length_validity = row["validity"]["bond_length"]
    spacegroup_validity = row["validity"]["spacegroup"]
    site_multiplicity_validity = row["validity"]["site_multiplicity"]
    valid = all(
        [
            formula_validity,
            bond_length_validity,
            spacegroup_validity,
            site_multiplicity_validity,
        ]
    )

    cif_sample = row["cif_string_sample"]
    xrd_q_continuous_sample = row["xrd_clean_sample"]["q"]
    xrd_iq_continuous_sample = row["xrd_clean_sample"]["iq"]
    xrd_q_discrete_sample = row["xrd_clean_sample"]["q_disc"]
    xrd_iq_discrete_sample = row["xrd_clean_sample"]["iq_disc"]

    cif_gen = row["cif_string_gen"]
    xrd_q_continuous_gen = row["xrd_clean_gen"]["q"]
    xrd_iq_continuous_gen = row["xrd_clean_gen"]["iq"]
    xrd_q_discrete_gen = row["xrd_clean_gen"]["q_disc"]
    xrd_iq_discrete_gen = row["xrd_clean_gen"]["iq_disc"]

    xrd_iq_discrete_sample_normed = xrd_iq_discrete_sample / np.sum(xrd_iq_discrete_sample)
    xrd_iq_discrete_gen_normed = xrd_iq_discrete_gen / np.sum(xrd_iq_discrete_gen)
    wd_value = wasserstein_distance(
        xrd_q_discrete_sample,
        xrd_q_discrete_gen,
        u_weights=xrd_iq_discrete_sample_normed,
        v_weights=xrd_iq_discrete_gen_normed,
    )

    rwp_value = residual_weighted_profile(xrd_iq_continuous_sample, xrd_iq_continuous_gen)
    rmsd_value = row["rmsd"]

    seq_len_sample = row["seq_len_sample"]
    seq_len_gen = row["seq_len_gen"]

    spacegroup_sym_sample, spacegroup_num_sample = spacegroup_info(cif_sample)
    spacegroup_sym_gen, spacegroup_num_gen = spacegroup_info(cif_gen)

    return {
        "rwp": rwp_value,
        "wd": wd_value,
        "rmsd": rmsd_value,
        "cif_sample": cif_sample,
        "xrd_q_discrete_sample": xrd_q_discrete_sample,
        "xrd_iq_discrete_sample": xrd_iq_discrete_sample,
        "xrd_q_continuous_sample": xrd_q_continuous_sample,
        "xrd_iq_continuous_sample": xrd_iq_continuous_sample,
        "spacegroup_sym_sample": spacegroup_sym_sample,
        "spacegroup_num_sample": spacegroup_num_sample,
        "seq_len_sample": seq_len_sample,
        "cif_gen": cif_gen,
        "xrd_q_discrete_gen": xrd_q_discrete_gen,
        "xrd_iq_discrete_gen": xrd_iq_discrete_gen,
        "xrd_q_continuous_gen": xrd_q_continuous_gen,
        "xrd_iq_continuous_gen": xrd_iq_continuous_gen,
        "seq_len_gen": seq_len_gen,
        "spacegroup_sym_gen": spacegroup_sym_gen,
        "spacegroup_num_gen": spacegroup_num_gen,
        "formula_validity": formula_validity,
        "spacegroup_validity": spacegroup_validity,
        "bond_length_validity": bond_length_validity,
        "site_multiplicity_validity": site_multiplicity_validity,
        "validity": valid,
    }
