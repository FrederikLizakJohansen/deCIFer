#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from typing import Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm.auto import tqdm

from decifer.bragg_refinement import (
    failed_refinement_record,
    load_bragg_refinement_config,
    run_bragg_refinement,
)
from decifer.decifer_dataset import DeciferDataset, h5_record_token_lengths
from decifer.decifer_model import Decifer, DeciferConfig
from decifer.evaluation_checkpoint import (
    EvaluationCheckpoint,
    checkpoint_paths,
    export_refinement_records,
    export_split_metrics,
    load_combined_checkpoint_frame,
)
from decifer.minicif import END_TOKEN, MinicifTokenizer, minicif_to_structure, parse_minicif
from decifer.minicif_v2 import (
    END_TOKEN as V2_END_TOKEN,
    MinicifV2Tokenizer,
    minicif_v2_to_structure,
    parse_minicif_v2,
)
from decifer.pxrd import (
    BraggArtifactSpec,
    bragg_artifact_batch,
    bragg_artifact_spec_from_config,
    clamp_qmax_for_wavelength,
    discrete_to_continuous_xrd,
    load_bragg_artifact_spec,
    nyquist_qstep,
    q_range_to_two_theta_range,
)
from bin.test_minicif_realtime import save_fit_figure
from bin.train import TrainConfig

PROMPT_MODE_ALIASES = {
    "pxrd": "start",
    "pxrd-elements": "constituents",
    "pxrd-elements-cs": "constituents-cs",
    "pxrd-elements-cs-sg": "constituents-cs-sg",
    "pxrd-stoichiometry": "formula",
    "pxrd-stoichiometry-cs": "formula-cs",
    "pxrd-stoichiometry-cs-sg": "formula-cs-sg",
}

CRYSTAL_SYSTEM_NAMES = {
    1: "triclinic",
    2: "monoclinic",
    3: "orthorhombic",
    4: "tetragonal",
    5: "trigonal",
    6: "hexagonal",
    7: "cubic",
}


def representation_api(tokenizer_name):
    if tokenizer_name == "minicif_v2":
        return MinicifV2Tokenizer(), parse_minicif_v2, minicif_v2_to_structure, V2_END_TOKEN
    return MinicifTokenizer(), parse_minicif, minicif_to_structure, END_TOKEN


def compatible_evaluation_indices(h5_path, config):
    block_size = config.get("block_size")
    if block_size is None:
        return None
    condition_tokens = (
        int(config.get("condition_n_tokens", 1))
        if config.get("condition") and not config.get("condition_cross_attention")
        else 0
    )
    max_record_length = int(block_size) + 1 - condition_tokens
    lengths = h5_record_token_lengths(h5_path)
    return np.flatnonzero(lengths <= max_record_length)


def rwp(reference, generated):
    reference = np.asarray(reference, dtype=float)
    generated = np.asarray(generated, dtype=float)
    return float(np.sqrt(np.sum((reference - generated) ** 2) / (np.sum(reference ** 2) + 1e-16)))


def profile_fit_statistics(observed, calculated):
    observed = np.asarray(observed, dtype=float)
    calculated = np.asarray(calculated, dtype=float)
    sigma = np.sqrt(np.maximum(observed, 0.0) + 1.0)
    weights = 1.0 / sigma**2
    residual = observed - calculated
    denominator = np.sum(weights * observed**2)
    return {
        "r_wp": float(
            np.sqrt(
                np.sum(weights * residual**2)
                / (denominator + 1e-16)
            )
        ),
        "chi_squared": float(np.mean((residual / sigma) ** 2)),
    }


def refined_structure_metrics(
    reference_structure,
    refined_structure,
    reference_space_group,
    reference_crystal_system,
    matcher,
    rmsd_threshold,
):
    rmsd = matcher.get_rms_dist(reference_structure, refined_structure)
    rmsd_value = None if rmsd is None else float(rmsd[0])
    match = rmsd_value is not None
    if rmsd_threshold > 0 and rmsd_value is not None:
        match = rmsd_value <= rmsd_threshold
    symmetry = SpacegroupAnalyzer(refined_structure)
    refined_space_group = int(symmetry.get_space_group_number())
    refined_crystal_system_name = symmetry.get_crystal_system()
    crystal_system_ids = {
        name: identifier for identifier, name in CRYSTAL_SYSTEM_NAMES.items()
    }
    refined_crystal_system = crystal_system_ids[refined_crystal_system_name]
    reference_elements = {
        element.symbol for element in reference_structure.composition.elements
    }
    refined_elements = {
        element.symbol for element in refined_structure.composition.elements
    }
    composition_match = (
        refined_structure.composition.reduced_formula
        == reference_structure.composition.reduced_formula
    )
    return {
        "refined_structure_ok": True,
        "refined_rmsd": rmsd_value,
        "refined_match": match,
        "refined_space_group": refined_space_group,
        "refined_crystal_system": refined_crystal_system,
        "refined_space_group_match": (
            refined_space_group == int(reference_space_group)
        ),
        "refined_crystal_system_match": (
            refined_crystal_system == int(reference_crystal_system)
        ),
        "refined_element_set_match": refined_elements == reference_elements,
        "refined_extra_elements": len(refined_elements - reference_elements),
        "refined_missing_elements": len(reference_elements - refined_elements),
        "refined_composition_match": composition_match,
        "refined_formula_match": composition_match,
    }


def load_checkpoint(path, device, use_best=True):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model_args = dict(checkpoint["model_args"])
    model = Decifer(DeciferConfig(**model_args)).to(device)
    state_key = "best_model_state" if use_best and checkpoint.get("best_model_state") is not None else "current_model"
    state_dict = checkpoint[state_key]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.device = device
    return checkpoint, model


def checkpoint_config(checkpoint):
    config = _config_to_dict(checkpoint.get("config"))
    metadata_config = _config_to_dict(checkpoint.get("run_metadata", {}).get("config"))
    merged = dict(metadata_config)
    merged.update(config)
    return merged


def _config_to_dict(config):
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "items"):
        return dict(config.items())
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return {}


def clean_xrd_kwargs(config, args):
    qmin = args.qmin if args.qmin is not None else float(config.get("qmin", 0.0))
    requested_qmax = args.qmax if args.qmax is not None else float(config.get("qmax", 10.0))
    wavelength = XRDCalculator(wavelength=args.wavelength).wavelength
    qmax = clamp_qmax_for_wavelength(requested_qmax, wavelength)
    if args.qstep is not None:
        qstep = args.qstep
    elif float(config.get("nyquist_points_per_fwhm", 0.0)) > 0:
        qstep = nyquist_qstep(float(config.get("fwhm_range_min", 0.05)), float(config["nyquist_points_per_fwhm"]))
    else:
        qstep = float(config.get("qstep", 0.01))
    fwhm = args.clean_fwhm
    if fwhm is None:
        fwhm = 0.5 * (float(config.get("fwhm_range_min", 0.05)) + float(config.get("fwhm_range_max", 0.05)))
    eta = args.eta
    if eta is None:
        eta = 0.5 * (float(config.get("eta_range_min", 0.5)) + float(config.get("eta_range_max", 0.5)))
    return {
        "qmin": qmin,
        "qmax": qmax,
        "qstep": qstep,
        "fwhm_range": (fwhm, fwhm),
        "eta_range": (eta, eta),
        "noise_range": None,
        "intensity_scale_range": None,
        "mask_prob": None,
        "final_normalize": bool(config.get("final_normalize_xrd", True)),
        "max_peaks": int(config.get("max_xrd_peaks", 0) or 0) or None,
    }


def dataset_path(dataset_dir, split):
    candidates = [
        os.path.join(dataset_dir, "serialized", f"{split}.h5"),
        os.path.join(dataset_dir, f"{split}.h5"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"could not find {split}.h5 under {dataset_dir}")


def prompt_from_minicif(minicif_string, mode, tokenizer):
    mode = PROMPT_MODE_ALIASES.get(mode, mode)
    fields = minicif_string.strip().split()
    if mode == "start":
        prompt = fields[0]
    elif mode == "constituents":
        if "formula" in fields:
            stop = fields.index("formula") + 1
        else:
            stop = next(i for i, field in enumerate(fields) if field.startswith("cs_"))
        prompt = " ".join(fields[:stop])
    elif mode == "constituents-cs":
        if "formula" in fields:
            raise ValueError(
                "crystal system cannot be supplied without stoichiometry in the sequential minicif_v2 grammar"
            )
        stop = next(i for i, field in enumerate(fields) if field.startswith("sg_"))
        prompt = " ".join(fields[:stop])
    elif mode == "constituents-cs-sg":
        if "formula" in fields:
            raise ValueError(
                "space group cannot be supplied without stoichiometry in the sequential minicif_v2 grammar"
            )
        stop = fields.index("cell")
        prompt = " ".join(fields[:stop])
    elif mode == "formula":
        stop = next(i for i, field in enumerate(fields) if field.startswith("cs_"))
        prompt = " ".join(fields[:stop])
    elif mode == "formula-cs":
        stop = next(i for i, field in enumerate(fields) if field.startswith("sg_"))
        prompt = " ".join(fields[:stop])
    elif mode == "formula-cs-sg":
        stop = fields.index("cell")
        prompt = " ".join(fields[:stop])
    else:
        raise ValueError(f"unknown prompt mode: {mode}")
    return torch.tensor(tokenizer.encode(tokenizer.tokenize_minicif(prompt)), dtype=torch.long)


def continuous_from_sparse(q, iq, xrd_kwargs):
    q_tensor = q if torch.is_tensor(q) else torch.tensor(q, dtype=torch.float32)
    iq_tensor = iq if torch.is_tensor(iq) else torch.tensor(iq, dtype=torch.float32)
    xrd = discrete_to_continuous_xrd(q_tensor.unsqueeze(0), iq_tensor.unsqueeze(0), **xrd_kwargs)
    return xrd["q"].cpu().numpy(), xrd["iq"][0].cpu().numpy(), xrd["iq"]


def condition_from_sparse(
    q,
    iq,
    xrd_kwargs,
    config,
    artifact_spec: Optional[BraggArtifactSpec] = None,
    artifact_device=None,
):
    q_tensor = q if torch.is_tensor(q) else torch.tensor(q, dtype=torch.float32)
    iq_tensor = iq if torch.is_tensor(iq) else torch.tensor(iq, dtype=torch.float32)
    encoder = config.get("condition_encoder", "mlp")
    _, reference_iq, _ = continuous_from_sparse(q_tensor, iq_tensor, xrd_kwargs)
    has_evaluation_artifacts = artifact_spec is not None
    artifact_spec = artifact_spec or bragg_artifact_spec_from_config(
        config, augment=False
    )
    device = artifact_device or q_tensor.device
    artifact_batch = bragg_artifact_batch(
        q_tensor.to(device).unsqueeze(0),
        iq_tensor.to(device).unsqueeze(0),
        spec=artifact_spec,
        qmin=xrd_kwargs["qmin"],
        qmax=xrd_kwargs["qmax"],
        qstep=xrd_kwargs["qstep"],
        include_dense=(
            has_evaluation_artifacts
            or encoder not in {"peak", "peak_fourier"}
        ),
        include_peaks=encoder in {"peak", "peak_fourier", "hybrid"},
        max_xrd_peaks=int(config.get("max_xrd_peaks", 0) or 0),
        max_peak_list_peaks=int(config.get("max_peak_list_peaks", 0) or 0),
    )
    observed_iq = (
        artifact_batch["iq"][0].detach().cpu().numpy()
        if has_evaluation_artifacts
        else reference_iq
    )
    if encoder in {"peak", "peak_fourier"}:
        return reference_iq, observed_iq, {
            "peak_q": artifact_batch["peak_q"],
            "peak_iq": artifact_batch["peak_iq"],
        }
    if encoder == "hybrid":
        return reference_iq, observed_iq, {
            "dense": artifact_batch["iq"],
            "peak_q": artifact_batch["peak_q"],
            "peak_iq": artifact_batch["peak_iq"],
        }
    return reference_iq, observed_iq, artifact_batch["iq"]


def structure_to_continuous_xrd(structure, xrd_kwargs, wavelength):
    calculator = XRDCalculator(wavelength=wavelength)
    _, two_theta_range = q_range_to_two_theta_range(xrd_kwargs["qmin"], xrd_kwargs["qmax"], calculator.wavelength)
    pattern = calculator.get_pattern(structure, two_theta_range=two_theta_range)
    theta = np.radians(pattern.x / 2)
    q_disc = torch.tensor(4 * np.pi * np.sin(theta) / calculator.wavelength, dtype=torch.float32)
    iq_disc = torch.tensor(pattern.y, dtype=torch.float32)
    iq_disc = iq_disc / (torch.max(iq_disc) + 1e-16)
    _, iq_cont, _ = continuous_from_sparse(q_disc, iq_disc, xrd_kwargs)
    return iq_cont


def save_evaluation_example(
    out_dir,
    split,
    source_sample_index,
    prompt_mode,
    xrd_kwargs,
    reference_iq,
    reference_structure,
    figure_rows,
    sample_name,
    figure_supercell,
):
    path = os.path.join(
        out_dir,
        "examples",
        split,
        f"sample_{source_sample_index:07d}_{prompt_mode}.png",
    )
    q_grid = (
        xrd_kwargs["qmin"]
        + np.arange(len(reference_iq)) * xrd_kwargs["qstep"]
    )
    save_fit_figure(
        path,
        q_grid,
        reference_iq,
        reference_structure,
        figure_rows,
        sample_name,
        figure_supercell,
    )
    return path


def repeat_condition(cond_vec, batch_size, device):
    if cond_vec is None:
        return None
    if isinstance(cond_vec, dict):
        return {key: value.to(device).repeat(batch_size, *([1] * (value.dim() - 1))) for key, value in cond_vec.items()}
    return cond_vec.to(device).repeat(batch_size, 1)


def generate_candidates(model, prompt, cond_vec, args, tokenizer):
    generated = []
    remaining = args.num_reps
    while remaining > 0:
        batch_size = min(args.generation_batch_size, remaining)
        batch_prompt = prompt.to(model.device).unsqueeze(0).repeat(batch_size, 1)
        batch_cond = repeat_condition(cond_vec, batch_size, model.device)
        batch = model.generate_batched_reps(
            batch_prompt,
            args.max_new_tokens,
            cond_vec=batch_cond,
            start_indices_batch=[[0]] * batch_size,
            temperature=args.temperature,
            top_k=args.top_k,
            disable_pbar=True,
            constrain_minicif=True,
            cfg_scale=args.cfg_scale,
        ).cpu().numpy()
        for ids in batch:
            ids = ids[ids != tokenizer.padding_id]
            generated.append(tokenizer.decode([int(token_id) for token_id in ids]))
        remaining -= batch_size
    return generated


def evaluate_split(
    split, h5_path, model, tokenizer, parse_fn, structure_fn, end_token,
    matcher, xrd_kwargs, config, args, artifact_spec=None, out_dir="",
    refinement_config=None, evaluation_checkpoint=None,
):
    compatible_indices = compatible_evaluation_indices(h5_path, config)
    dataset = DeciferDataset(
        h5_path,
        ["cif_name", "minicif_string", "cif_tokens", "xrd.q", "xrd.iq", "spacegroup", "crystal_system"],
        indices=compatible_indices,
    )
    if compatible_indices is not None:
        n_total = len(h5_record_token_lengths(h5_path))
        n_excluded = n_total - len(compatible_indices)
        if n_excluded:
            print(
                f"{h5_path}: excluding {n_excluded}/{n_total} reference structures "
                "that exceed the checkpoint context window.",
                flush=True,
            )
    n_items = len(dataset) if args.max_items <= 0 else min(args.max_items, len(dataset))
    source_indices = (
        np.arange(n_items, dtype=np.int64)
        if dataset.indices is None
        else dataset.indices[:n_items]
    )
    crystal_system_values = np.asarray(dataset.data["crystal_system"])[source_indices]
    available_crystal_systems = [
        crystal_system
        for crystal_system in CRYSTAL_SYSTEM_NAMES
        if crystal_system in set(int(value) for value in crystal_system_values)
    ]
    example_targets = set(available_crystal_systems[:args.plot_examples])
    plotted_examples = {mode: set() for mode in args.prompt_modes}
    plotted_example_counts = {mode: 0 for mode in args.prompt_modes}
    completed_units = evaluation_checkpoint.completed_units()
    if completed_units:
        print(
            f"{h5_path}: resuming with {len(completed_units)} completed "
            "sample/prompt units.",
            flush=True,
        )
    for sample_index in tqdm(range(n_items), desc=f"Evaluating {split}"):
        source_sample_index = dataset.source_index(sample_index)
        pending_prompt_modes = [
            mode
            for mode in args.prompt_modes
            if (source_sample_index, mode) not in completed_units
        ]
        if not pending_prompt_modes:
            continue
        item = dataset[sample_index]
        reference_minicif = item["minicif_string"]
        try:
            reference_parsed = parse_fn(reference_minicif)
            reference_structure = structure_fn(reference_minicif)
            sample_artifact_spec = artifact_spec
            if artifact_spec is not None and artifact_spec.artifacts.seed is not None:
                sample_artifact_spec = replace(
                    artifact_spec,
                    artifacts=replace(
                        artifact_spec.artifacts,
                        seed=artifact_spec.artifacts.seed + source_sample_index,
                    ),
                )
            reference_iq, observed_iq, cond = condition_from_sparse(
                item["xrd.q"],
                item["xrd.iq"],
                xrd_kwargs,
                config,
                artifact_spec=sample_artifact_spec,
                artifact_device=model.device,
            )
        except Exception as exc:
            for prompt_mode in pending_prompt_modes:
                evaluation_checkpoint.save_unit(
                    source_sample_index,
                    prompt_mode,
                    [{
                        "split": split,
                        "sample_index": source_sample_index,
                        "cif_name": item["cif_name"],
                        "rep": -1,
                        "prompt_mode": prompt_mode,
                        "reference_error": str(exc),
                        "parse_ok": False,
                        "match": False,
                    }],
                    [],
                )
            continue

        for prompt_mode in pending_prompt_modes:
            prompt = prompt_from_minicif(reference_minicif, prompt_mode, tokenizer)
            candidates = generate_candidates(model, prompt, cond, args, tokenizer)
            mode_rows = []
            mode_refinement_records = []
            figure_rows = []
            reference_crystal_system = int(reference_parsed.crystal_system)
            missing_target = (
                reference_crystal_system in example_targets
                and reference_crystal_system not in plotted_examples[prompt_mode]
            )
            targets_complete = example_targets.issubset(
                plotted_examples[prompt_mode]
            )
            plot_example = args.plot_examples > 0 and (
                missing_target
                or (
                    targets_complete
                    and plotted_example_counts[prompt_mode] < args.plot_examples
                )
            )
            for rep, generated_minicif in enumerate(candidates):
                refinement_identity = {
                    "split": split,
                    "sample_index": source_sample_index,
                    "cif_name": item["cif_name"],
                    "prompt_mode": prompt_mode,
                    "rep": rep,
                    "generated_minicif": generated_minicif,
                }
                refinement_record = None
                row = {
                    "split": split,
                    "sample_index": source_sample_index,
                    "cif_name": item["cif_name"],
                    "prompt_mode": prompt_mode,
                    "rep": rep,
                    "reference_minicif": reference_minicif,
                    "generated_minicif": generated_minicif,
                    "generated_n_tokens": len(tokenizer.tokenize_minicif(generated_minicif)),
                    "finished": generated_minicif.strip().endswith(end_token),
                    "reference_space_group": reference_parsed.space_group,
                    "reference_crystal_system": reference_parsed.crystal_system,
                    "parse_ok": False,
                    "structure_ok": False,
                    "match": False,
                    "refinement_requested": bool(
                        refinement_config is not None
                        and refinement_config.enabled
                    ),
                }
                try:
                    generated_parsed = parse_fn(generated_minicif)
                    row.update({
                        "parse_ok": True,
                        "generated_space_group": generated_parsed.space_group,
                        "generated_crystal_system": generated_parsed.crystal_system,
                        "space_group_match": generated_parsed.space_group == reference_parsed.space_group,
                        "crystal_system_match": generated_parsed.crystal_system == reference_parsed.crystal_system,
                        "element_set_match": set(generated_parsed.elements) == set(reference_parsed.elements),
                        "extra_elements": len(set(generated_parsed.elements) - set(reference_parsed.elements)),
                        "missing_elements": len(set(reference_parsed.elements) - set(generated_parsed.elements)),
                    })
                    if hasattr(reference_parsed, "formula"):
                        row["formula_match"] = generated_parsed.formula == reference_parsed.formula
                    generated_structure = structure_fn(generated_minicif)
                    generated_iq = structure_to_continuous_xrd(generated_structure, xrd_kwargs, args.wavelength)
                    rmsd = matcher.get_rms_dist(reference_structure, generated_structure)
                    rmsd_value = None if rmsd is None else float(rmsd[0])
                    match = rmsd_value is not None
                    if args.rmsd_threshold > 0 and rmsd_value is not None:
                        match = rmsd_value <= args.rmsd_threshold
                    row.update({
                        "structure_ok": True,
                        "rwp": rwp(reference_iq, generated_iq),
                        "rmsd": rmsd_value,
                        "match": match,
                        "generated_formula": generated_structure.composition.reduced_formula,
                        "composition_match": generated_structure.composition.reduced_formula == reference_structure.composition.reduced_formula,
                    })
                    if plot_example:
                        figure_rows.append({
                            "rep": rep,
                            "rwp": row["rwp"],
                            "generated_iq": generated_iq,
                            "generated_structure": generated_structure,
                        })
                    if refinement_config is not None and refinement_config.enabled:
                        initial_fit_statistics = profile_fit_statistics(
                            observed_iq, generated_iq
                        )
                        initial_fit_statistics["evaluation_r_wp"] = row["rwp"]
                        initial_metrics = {
                            "evaluation_r_wp": row["rwp"],
                            "rmsd": row["rmsd"],
                            "match": row["match"],
                            "composition_match": row["composition_match"],
                            "space_group_match": row["space_group_match"],
                            "crystal_system_match": row["crystal_system_match"],
                            "element_set_match": row["element_set_match"],
                        }
                        refinement_result, refinement_record = run_bragg_refinement(
                            xrd_kwargs["qmin"]
                            + np.arange(len(observed_iq)) * xrd_kwargs["qstep"],
                            observed_iq,
                            generated_structure,
                            refinement_config,
                            initial_fit_statistics=initial_fit_statistics,
                        )
                        row["refinement_attempted"] = True
                        row["refinement_succeeded"] = (
                            refinement_result is not None
                        )
                        if refinement_result is None:
                            error = refinement_record["error"]
                            row["refinement_error"] = (
                                f"{error['type']}: {error['message']}"
                            )
                        else:
                            fit_statistics = refinement_result.fit_statistics
                            row.update({
                                "refinement_initial_r_wp": (
                                    initial_fit_statistics["r_wp"]
                                ),
                                "refinement_initial_chi_squared": (
                                    initial_fit_statistics["chi_squared"]
                                ),
                                "refinement_status": refinement_result.status,
                                "refinement_converged": (
                                    refinement_result.status == "converged"
                                ),
                                "refinement_convergence_classification": (
                                    refinement_result.convergence.get(
                                        "classification"
                                    )
                                ),
                                "refinement_r_wp": fit_statistics.get("r_wp"),
                                "refinement_chi_squared": fit_statistics.get(
                                    "chi_squared"
                                ),
                                "refinement_held_out_r_wp": fit_statistics.get(
                                    "held_out_r_wp"
                                ),
                                "refinement_warning_count": len(
                                    refinement_result.warnings
                                ),
                                "refinement_parameter_count": len(
                                    refinement_result.parameters
                                ),
                            })
                            try:
                                refined_structure = (
                                    refinement_result.refined_structure
                                )
                                refined_iq = structure_to_continuous_xrd(
                                    refined_structure,
                                    xrd_kwargs,
                                    args.wavelength,
                                )
                                row["refined_rwp"] = rwp(
                                    reference_iq, refined_iq
                                )
                                row["refinement_improved_rwp"] = (
                                    row["refined_rwp"] < row["rwp"]
                                )
                                row.update(refined_structure_metrics(
                                    reference_structure,
                                    refined_structure,
                                    reference_parsed.space_group,
                                    reference_parsed.crystal_system,
                                    matcher,
                                    args.rmsd_threshold,
                                ))
                                refinement_record["refined_fit_statistics"][
                                    "evaluation_r_wp"
                                ] = row["refined_rwp"]
                                refinement_record["evaluation_metrics"] = {
                                    "initial": initial_metrics,
                                    "refined": {
                                        key: row[key]
                                        for key in (
                                            "refined_rwp",
                                            "refined_rmsd",
                                            "refined_match",
                                            "refined_composition_match",
                                            "refined_space_group_match",
                                            "refined_crystal_system_match",
                                            "refined_element_set_match",
                                        )
                                    },
                                }
                                if plot_example:
                                    figure_rows.append({
                                        "rep": f"refined {rep}",
                                        "rwp": row["refined_rwp"],
                                        "generated_iq": refined_iq,
                                        "generated_structure": refined_structure,
                                    })
                            except Exception as exc:
                                row["refined_metric_error"] = str(exc)
                                refinement_record["evaluation_metric_error"] = {
                                    "type": type(exc).__name__,
                                    "message": str(exc),
                                }
                except Exception as exc:
                    row["error"] = str(exc)
                    if refinement_config is not None and refinement_config.enabled:
                        refinement_record = failed_refinement_record(
                            refinement_config,
                            exc,
                            initial_fit_statistics={
                                "evaluation_r_wp": row.get("rwp")
                            },
                        )
                        row["refinement_attempted"] = False
                        row["refinement_succeeded"] = False
                        row["refinement_error"] = (
                            f"InvalidCandidate: {exc}"
                        )
                if refinement_record is not None:
                    mode_refinement_records.append(
                        {**refinement_identity, **refinement_record}
                    )
                mode_rows.append(row)

            if plot_example and figure_rows:
                save_evaluation_example(
                    out_dir,
                    split,
                    source_sample_index,
                    prompt_mode,
                    xrd_kwargs,
                    reference_iq,
                    reference_structure,
                    figure_rows,
                    (
                        f"{item['cif_name']} | {prompt_mode} | "
                        f"{CRYSTAL_SYSTEM_NAMES.get(reference_crystal_system, reference_crystal_system)}"
                    ),
                    args.figure_supercell,
                )
                plotted_examples[prompt_mode].add(reference_crystal_system)
                plotted_example_counts[prompt_mode] += 1
            evaluation_checkpoint.save_unit(
                source_sample_index,
                prompt_mode,
                mode_rows,
                mode_refinement_records,
            )


def summarize(df):
    if df.empty or "split" not in df.columns:
        return pd.DataFrame()
    summaries = []
    group_cols = ["split"]
    if "prompt_mode" in df.columns:
        group_cols.append("prompt_mode")
    for group_key, split_df in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        valid_rwp = split_df.dropna(subset=["rwp"]) if "rwp" in split_df else split_df.iloc[0:0]
        by_sample = split_df.groupby("sample_index")
        candidates = split_df[split_df["rep"] >= 0]
        candidate_samples = candidates.groupby("sample_index")
        best_rwp = by_sample["rwp"].min() if "rwp" in split_df else pd.Series(dtype=float)
        if "refined_rwp" in split_df:
            valid_refined_rwp = split_df.dropna(subset=["refined_rwp"])
            best_refined_rwp = by_sample["refined_rwp"].min()
        else:
            valid_refined_rwp = split_df.iloc[0:0]
            best_refined_rwp = pd.Series(dtype=float)
        summary = {
            "split": group_key[0],
            "n_samples": int(split_df["sample_index"].nunique()),
            "n_candidates": int(len(split_df[split_df["rep"] >= 0])),
            "parse_rate": float(split_df["parse_ok"].fillna(False).mean()),
            "valid_minicif_rate": float(split_df["parse_ok"].fillna(False).mean()),
            "structure_rate": float(split_df["structure_ok"].fillna(False).mean()) if "structure_ok" in split_df else np.nan,
            "finish_rate": float(split_df["finished"].fillna(False).mean()) if "finished" in split_df else np.nan,
            "candidate_match_rate": float(split_df["match"].fillna(False).mean()),
            "best_of_k_match_rate": float(by_sample["match"].max().fillna(False).mean()),
            "median_rwp": float(valid_rwp["rwp"].median()) if not valid_rwp.empty else np.nan,
            "median_best_rwp": float(best_rwp.median()) if not best_rwp.empty else np.nan,
            "mean_best_rwp": float(best_rwp.mean()) if not best_rwp.empty else np.nan,
            "median_best_refined_rwp": float(best_refined_rwp.median()) if not best_refined_rwp.empty else np.nan,
            "mean_best_refined_rwp": float(best_refined_rwp.mean()) if not best_refined_rwp.empty else np.nan,
            "median_refined_rwp": float(valid_refined_rwp["refined_rwp"].median()) if not valid_refined_rwp.empty else np.nan,
            "median_generated_n_tokens": float(split_df["generated_n_tokens"].dropna().median()) if "generated_n_tokens" in split_df else np.nan,
            "median_matched_rmsd": float(split_df.loc[split_df["match"] == True, "rmsd"].median()) if "rmsd" in split_df else np.nan,
            "space_group_accuracy": float(split_df["space_group_match"].fillna(False).mean()) if "space_group_match" in split_df else np.nan,
            "crystal_system_accuracy": float(split_df["crystal_system_match"].fillna(False).mean()) if "crystal_system_match" in split_df else np.nan,
            "element_set_accuracy": float(split_df["element_set_match"].fillna(False).mean()) if "element_set_match" in split_df else np.nan,
            "mean_extra_elements": float(split_df["extra_elements"].dropna().mean()) if "extra_elements" in split_df else np.nan,
            "mean_missing_elements": float(split_df["missing_elements"].dropna().mean()) if "missing_elements" in split_df else np.nan,
            "mean_unique_minicif_fraction": _mean_unique_fraction(
                candidate_samples, "generated_minicif"
            ),
            "mean_unique_formulas_per_sample": _mean_group_nunique(
                candidate_samples, "generated_formula"
            ),
            "mean_unique_space_groups_per_sample": _mean_group_nunique(
                candidate_samples, "generated_space_group"
            ),
            "mean_unique_crystal_systems_per_sample": _mean_group_nunique(
                candidate_samples, "generated_crystal_system"
            ),
            "composition_match_rate": float(split_df["composition_match"].fillna(False).mean()) if "composition_match" in split_df else np.nan,
            "formula_accuracy": float(split_df["formula_match"].fillna(False).mean()) if "formula_match" in split_df else np.nan,
            "refinement_success_rate": float(split_df["refinement_succeeded"].fillna(False).mean()) if "refinement_succeeded" in split_df else np.nan,
            "refinement_rwp_improvement_rate": float(split_df["refinement_improved_rwp"].dropna().mean()) if "refinement_improved_rwp" in split_df else np.nan,
            "refined_structure_rate": float(split_df["refined_structure_ok"].fillna(False).mean()) if "refined_structure_ok" in split_df else np.nan,
            "refined_candidate_match_rate": float(split_df["refined_match"].fillna(False).mean()) if "refined_match" in split_df else np.nan,
            "refined_best_of_k_match_rate": float(by_sample["refined_match"].max().fillna(False).mean()) if "refined_match" in split_df else np.nan,
            "refined_space_group_accuracy": float(split_df["refined_space_group_match"].fillna(False).mean()) if "refined_space_group_match" in split_df else np.nan,
            "refined_crystal_system_accuracy": float(split_df["refined_crystal_system_match"].fillna(False).mean()) if "refined_crystal_system_match" in split_df else np.nan,
            "refined_element_set_accuracy": float(split_df["refined_element_set_match"].fillna(False).mean()) if "refined_element_set_match" in split_df else np.nan,
            "refined_composition_match_rate": float(split_df["refined_composition_match"].fillna(False).mean()) if "refined_composition_match" in split_df else np.nan,
            "refined_formula_accuracy": float(split_df["refined_formula_match"].fillna(False).mean()) if "refined_formula_match" in split_df else np.nan,
            "mean_refined_extra_elements": float(split_df["refined_extra_elements"].dropna().mean()) if "refined_extra_elements" in split_df else np.nan,
            "mean_refined_missing_elements": float(split_df["refined_missing_elements"].dropna().mean()) if "refined_missing_elements" in split_df else np.nan,
            "median_refined_matched_rmsd": float(split_df.loc[split_df["refined_match"] == True, "refined_rmsd"].median()) if "refined_rmsd" in split_df and "refined_match" in split_df else np.nan,
        }
        if len(group_key) > 1:
            summary["prompt_mode"] = group_key[1]
        summaries.append(summary)
    return pd.DataFrame(summaries)


def _mean_unique_fraction(grouped, column):
    if column not in grouped.obj or grouped.ngroups == 0:
        return np.nan
    values = grouped[column].agg(
        lambda series: series.dropna().nunique() / max(series.notna().sum(), 1)
    )
    return float(values.mean()) if not values.empty else np.nan


def _mean_group_nunique(grouped, column):
    if column not in grouped.obj or grouped.ngroups == 0:
        return np.nan
    values = grouped[column].nunique(dropna=True)
    return float(values.mean()) if not values.empty else np.nan


def summarize_by_crystal_system(df):
    required = {"split", "sample_index", "reference_crystal_system"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()
    group_cols = ["split"]
    if "prompt_mode" in df.columns:
        group_cols.append("prompt_mode")
    rows = []
    for group_key, group_df in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        for crystal_system, system_df in group_df.groupby(
            "reference_crystal_system", dropna=True
        ):
            samples = []
            for _, sample_df in system_df.groupby("sample_index"):
                sample = {}
                for source, target in (
                    ("parse_ok", "valid_minicif_rate"),
                    ("structure_ok", "structure_rate"),
                    ("match", "best_of_k_match_rate"),
                    ("element_set_match", "element_set_accuracy"),
                    ("composition_match", "composition_match_rate"),
                    ("space_group_match", "space_group_accuracy"),
                    ("crystal_system_match", "crystal_system_accuracy"),
                    ("refinement_succeeded", "refinement_success_rate"),
                    ("refined_structure_ok", "refined_structure_rate"),
                    ("refined_match", "refined_best_of_k_match_rate"),
                    ("refined_element_set_match", "refined_element_set_accuracy"),
                    ("refined_composition_match", "refined_composition_match_rate"),
                    ("refined_space_group_match", "refined_space_group_accuracy"),
                    ("refined_crystal_system_match", "refined_crystal_system_accuracy"),
                ):
                    sample[target] = (
                        bool(sample_df[source].fillna(False).any())
                        if source in sample_df
                        else np.nan
                    )
                sample["best_rwp"] = (
                    float(sample_df["rwp"].dropna().min())
                    if "rwp" in sample_df and not sample_df["rwp"].dropna().empty
                    else np.nan
                )
                sample["best_refined_rwp"] = (
                    float(sample_df["refined_rwp"].dropna().min())
                    if "refined_rwp" in sample_df
                    and not sample_df["refined_rwp"].dropna().empty
                    else np.nan
                )
                samples.append(sample)
            sample_df = pd.DataFrame(samples)
            row = {
                "split": group_key[0],
                "reference_crystal_system": int(crystal_system),
                "crystal_system_name": CRYSTAL_SYSTEM_NAMES.get(
                    int(crystal_system), str(int(crystal_system))
                ),
                "n_samples": len(sample_df),
            }
            if len(group_key) > 1:
                row["prompt_mode"] = group_key[1]
            for metric in (
                "valid_minicif_rate",
                "structure_rate",
                "best_of_k_match_rate",
                "element_set_accuracy",
                "composition_match_rate",
                "space_group_accuracy",
                "crystal_system_accuracy",
                "refinement_success_rate",
                "refined_structure_rate",
                "refined_best_of_k_match_rate",
                "refined_element_set_accuracy",
                "refined_composition_match_rate",
                "refined_space_group_accuracy",
                "refined_crystal_system_accuracy",
            ):
                row[metric] = float(sample_df[metric].mean())
            row["median_best_rwp"] = float(sample_df["best_rwp"].median())
            row["mean_best_rwp"] = float(sample_df["best_rwp"].mean())
            row["median_best_refined_rwp"] = float(
                sample_df["best_refined_rwp"].median()
            )
            row["mean_best_refined_rwp"] = float(
                sample_df["best_refined_rwp"].mean()
            )
            rows.append(row)
    return pd.DataFrame(rows)


def plot_learning_curves(checkpoint, out_dir):
    metrics = checkpoint.get("training_metrics") or {}
    epochs = metrics.get("epochs") or list(range(len(metrics.get("train_losses", []))))
    train_losses = metrics.get("train_losses", [])
    val_losses = metrics.get("val_losses", [])
    if not train_losses and not val_losses:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    if train_losses:
        ax.plot(epochs[:len(train_losses)], train_losses, label="train")
    if val_losses:
        ax.plot(epochs[:len(val_losses)], val_losses, label="validation")
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "learning_curves.png"))
    plt.close(fig)


def plot_metric_summary(summary, out_dir):
    if summary.empty:
        return
    metrics = [
        "valid_minicif_rate",
        "structure_rate",
        "best_of_k_match_rate",
        "candidate_match_rate",
        "element_set_accuracy",
        "composition_match_rate",
        "space_group_accuracy",
        "crystal_system_accuracy",
    ]
    available = [metric for metric in metrics if metric in summary.columns]
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
    x = np.arange(len(summary))
    labels = summary["split"].astype(str)
    if "prompt_mode" in summary.columns:
        labels = labels + "/" + summary["prompt_mode"].astype(str)
    width = 0.8 / max(1, len(available))
    for i, metric in enumerate(available):
        ax.bar(x + i * width, summary[metric], width=width, label=metric)
    ax.set_xticks(x + width * (len(available) - 1) / 2)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("rate")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "metric_summary.png"))
    plt.close(fig)


def plot_refinement_metric_comparison(summary, out_dir):
    if (
        summary.empty
        or "refinement_success_rate" not in summary
        or summary["refinement_success_rate"].dropna().empty
    ):
        return
    metrics = [
        (
            "best_of_k_match_rate",
            "refined_best_of_k_match_rate",
            "structure match",
        ),
        (
            "element_set_accuracy",
            "refined_element_set_accuracy",
            "element set",
        ),
        (
            "composition_match_rate",
            "refined_composition_match_rate",
            "composition",
        ),
        (
            "crystal_system_accuracy",
            "refined_crystal_system_accuracy",
            "crystal system",
        ),
        (
            "space_group_accuracy",
            "refined_space_group_accuracy",
            "space group",
        ),
    ]
    labels = summary["split"].astype(str)
    if "prompt_mode" in summary.columns:
        labels = labels + "/" + summary["prompt_mode"].astype(str)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 8),
        dpi=160,
        squeeze=False,
        constrained_layout=True,
    )
    x = np.arange(len(summary))
    for ax, (initial, refined, title) in zip(axes.flat, metrics):
        ax.bar(x - 0.2, summary[initial], width=0.4, label="initial")
        ax.bar(x + 0.2, summary[refined], width=0.4, label="refined")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    success_ax = axes.flat[-1]
    success_ax.bar(x, summary["refinement_success_rate"], width=0.6)
    success_ax.set_title("refinement success")
    success_ax.set_ylim(0, 1)
    success_ax.set_xticks(x)
    success_ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    success_ax.grid(axis="y", alpha=0.25)
    axes.flat[0].legend(frameon=False)
    fig.savefig(os.path.join(out_dir, "refinement_metric_comparison.png"))
    plt.close(fig)


def plot_rwp_distribution(df, out_dir):
    if "rwp" not in df or df["rwp"].dropna().empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    group_cols = ["split"] + (
        ["prompt_mode"] if "prompt_mode" in df.columns else []
    )
    groups = list(df.groupby(group_cols, dropna=False))
    values = []
    labels = []
    for key, group in groups:
        group_label = _metric_group_label(key)
        for column, state in (("rwp", "initial"), ("refined_rwp", "refined")):
            if column in group and not group[column].dropna().empty:
                values.append(group[column].dropna().to_numpy())
                labels.append(f"{group_label}\n{state}")
    ax.boxplot(values, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Rwp")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rwp_distribution.png"))
    plt.close(fig)


def _metric_group_label(key):
    values = key if isinstance(key, tuple) else (key,)
    return "/".join(str(value) for value in values)


def plot_best_rwp_cdf(df, out_dir):
    if "rwp" not in df or df["rwp"].dropna().empty:
        return
    group_cols = ["split"] + (
        ["prompt_mode"] if "prompt_mode" in df.columns else []
    )
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for key, group in df.groupby(group_cols, dropna=False):
        for column, state in (("rwp", "initial"), ("refined_rwp", "refined")):
            if column not in group:
                continue
            best = (
                group.groupby("sample_index")[column]
                .min()
                .dropna()
                .sort_values()
            )
            if best.empty:
                continue
            fraction = np.arange(1, len(best) + 1) / len(best)
            ax.step(
                best.to_numpy(),
                fraction,
                where="post",
                label=f"{_metric_group_label(key)}/{state}",
            )
    ax.set_xlabel("best-of-k Rwp")
    ax.set_ylabel("fraction of samples")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "best_rwp_cdf.png"))
    plt.close(fig)


def _crystal_system_matrix(summary, metric):
    group_cols = ["split"] + (
        ["prompt_mode"] if "prompt_mode" in summary.columns else []
    )
    groups = list(summary.groupby(group_cols, dropna=False))
    labels = [_metric_group_label(key) for key, _ in groups]
    matrix = np.full((len(groups), len(CRYSTAL_SYSTEM_NAMES)), np.nan)
    for row_index, (_, group) in enumerate(groups):
        for _, row in group.iterrows():
            crystal_system = int(row["reference_crystal_system"])
            if crystal_system in CRYSTAL_SYSTEM_NAMES:
                matrix[row_index, crystal_system - 1] = row[metric]
    return matrix, labels


def _plot_crystal_system_metric_panels(summary, metrics, path):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 8),
        dpi=160,
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for panel_index, (ax, (metric, title)) in enumerate(zip(axes.flat, metrics)):
        matrix, labels = _crystal_system_matrix(summary, metric)
        image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(range(7))
        ax.set_xticklabels(
            ["tri", "mono", "ortho", "tetra", "trig", "hexa", "cubic"],
            rotation=45,
            ha="right",
        )
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if panel_index % 3 == 0 else [], fontsize=7)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if np.isfinite(value):
                    ax.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black" if value > 0.65 else "white",
                    )
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="best-of-k sample rate")
    fig.savefig(path)
    plt.close(fig)


def plot_crystal_system_metrics(summary, out_dir):
    if summary.empty:
        return
    _plot_crystal_system_metric_panels(
        summary,
        [
            ("structure_rate", "valid structure"),
            ("element_set_accuracy", "element set"),
            ("composition_match_rate", "composition"),
            ("crystal_system_accuracy", "crystal system"),
            ("space_group_accuracy", "space group"),
            ("best_of_k_match_rate", "structure match"),
        ],
        os.path.join(out_dir, "crystal_system_metrics.png"),
    )
    if (
        "refinement_success_rate" in summary
        and not summary["refinement_success_rate"].dropna().empty
    ):
        _plot_crystal_system_metric_panels(
            summary,
            [
                ("refined_structure_rate", "refined structure"),
                ("refined_element_set_accuracy", "refined element set"),
                (
                    "refined_composition_match_rate",
                    "refined composition",
                ),
                (
                    "refined_crystal_system_accuracy",
                    "refined crystal system",
                ),
                (
                    "refined_space_group_accuracy",
                    "refined space group",
                ),
                (
                    "refined_best_of_k_match_rate",
                    "refined structure match",
                ),
            ],
            os.path.join(out_dir, "crystal_system_refined_metrics.png"),
        )


def plot_crystal_system_rwp(summary, out_dir):
    if summary.empty or summary["median_best_rwp"].dropna().empty:
        return
    has_refined = (
        "median_best_refined_rwp" in summary
        and not summary["median_best_refined_rwp"].dropna().empty
    )
    columns = [
        ("median_best_rwp", "initial"),
        ("median_best_refined_rwp", "refined"),
    ] if has_refined else [("median_best_rwp", "initial")]
    fig, axes = plt.subplots(
        1,
        len(columns),
        figsize=(9 * len(columns), max(4, 0.45 * len(summary) + 2)),
        dpi=160,
        squeeze=False,
        constrained_layout=True,
    )
    matrices = [
        (*_crystal_system_matrix(summary, column), title)
        for column, title in columns
    ]
    finite = np.concatenate(
        [matrix[np.isfinite(matrix)] for matrix, _, _ in matrices]
    )
    vmin, vmax = float(finite.min()), float(finite.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12
    for ax, (matrix, labels, title) in zip(axes.flat, matrices):
        image = ax.imshow(
            matrix,
            aspect="auto",
            cmap="magma_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xticks(range(7))
        ax.set_xticklabels(
            CRYSTAL_SYSTEM_NAMES.values(),
            rotation=35,
            ha="right",
        )
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(f"{title} median best-of-k Rwp")
    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label="median best-of-k Rwp",
    )
    fig.savefig(os.path.join(out_dir, "crystal_system_rwp.png"))
    plt.close(fig)


def plot_crystal_system_rwp_distribution(df, out_dir):
    required = {"rwp", "reference_crystal_system"}
    if df.empty or not required.issubset(df.columns):
        return
    group_cols = ["split"] + (
        ["prompt_mode"] if "prompt_mode" in df.columns else []
    )
    groups = list(df.groupby(group_cols, dropna=False))
    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(12, max(4.5, 4.2 * len(groups))),
        dpi=160,
        squeeze=False,
    )
    for ax, (key, group) in zip(axes.flat, groups):
        positions = []
        values = []
        colors = []
        for crystal_system in CRYSTAL_SYSTEM_NAMES:
            system = group[
                group["reference_crystal_system"] == crystal_system
            ]
            initial = system["rwp"].dropna().to_numpy()
            if initial.size:
                positions.append(crystal_system - 0.18)
                values.append(initial)
                colors.append("#0072B2")
            if "refined_rwp" in system:
                refined = system["refined_rwp"].dropna().to_numpy()
                if refined.size:
                    positions.append(crystal_system + 0.18)
                    values.append(refined)
                    colors.append("#D55E00")
        if values:
            artists = ax.boxplot(
                values,
                positions=positions,
                widths=0.3,
                showfliers=False,
                patch_artist=True,
            )
            for patch, color in zip(artists["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.65)
        ax.plot([], [], color="#0072B2", linewidth=8, alpha=0.65, label="initial")
        if "refined_rwp" in group:
            ax.plot([], [], color="#D55E00", linewidth=8, alpha=0.65, label="refined")
        ax.set_xticks(range(1, 8))
        ax.set_xticklabels(
            CRYSTAL_SYSTEM_NAMES.values(),
            rotation=25,
            ha="right",
        )
        ax.set_ylabel("Rwp")
        ax.set_title(_metric_group_label(key))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "crystal_system_rwp_distribution.png"))
    plt.close(fig)


def plot_rwp_vs_rmsd(df, out_dir):
    if not {"rwp", "rmsd"}.issubset(df.columns):
        return
    valid = df.dropna(subset=["rwp", "rmsd"])
    if valid.empty:
        return
    group_cols = ["split"] + (
        ["prompt_mode"] if "prompt_mode" in valid.columns else []
    )
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    for key, group in valid.groupby(group_cols, dropna=False):
        ax.scatter(
            group["rwp"],
            group["rmsd"],
            s=13,
            alpha=0.35,
            label=f"{_metric_group_label(key)}/initial",
        )
    if {"refined_rwp", "refined_rmsd"}.issubset(df.columns):
        refined = df.dropna(subset=["refined_rwp", "refined_rmsd"])
        for key, group in refined.groupby(group_cols, dropna=False):
            ax.scatter(
                group["refined_rwp"],
                group["refined_rmsd"],
                s=18,
                marker="x",
                alpha=0.5,
                label=f"{_metric_group_label(key)}/refined",
            )
    ax.set_xlabel("Rwp")
    ax.set_ylabel("matched structure RMSD")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rwp_vs_rmsd.png"))
    plt.close(fig)


def file_identity(path, *, content_hash=False):
    path = os.path.abspath(path)
    stat = os.stat(path)
    identity = {
        "path": path,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if content_hash:
        with open(path, "rb") as handle:
            identity["sha256"] = hashlib.sha256(handle.read()).hexdigest()
    return identity


def evaluation_signature(
    args,
    checkpoint_path,
    h5_path,
    tokenizer_name,
    xrd_kwargs,
    refinement_config,
):
    return {
        "schema_version": 1,
        "checkpoint": file_identity(checkpoint_path),
        "dataset": file_identity(h5_path),
        "tokenizer": tokenizer_name,
        "use_current": args.use_current,
        "num_reps": args.num_reps,
        "generation_batch_size": args.generation_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "cfg_scale": args.cfg_scale,
        "rmsd_threshold": args.rmsd_threshold,
        "seed": args.seed,
        "wavelength": args.wavelength,
        "xrd_kwargs": xrd_kwargs,
        "artifact_config": (
            file_identity(args.artifact_config, content_hash=True)
            if args.artifact_config
            else None
        ),
        "refinement": refinement_config.to_dict(),
    }


def write_combined_report(out_dir, report_metadata=None, checkpoint=None):
    paths = checkpoint_paths(out_dir)
    if not paths:
        raise ValueError(
            f"no evaluation checkpoints found under {out_dir}"
        )
    results = load_combined_checkpoint_frame(paths)
    summary = summarize(results)
    crystal_system_summary = summarize_by_crystal_system(results)
    export_split_metrics(paths, out_dir)

    refinement_output = os.path.join(
        out_dir, "minicif_refinement_results.jsonl.gz"
    )
    refinement_count = export_refinement_records(paths, refinement_output)
    if refinement_count == 0:
        os.unlink(refinement_output)
        refinement_output = None

    results.to_csv(
        os.path.join(out_dir, "minicif_generation_metrics.csv"),
        index=False,
    )
    summary.to_csv(os.path.join(out_dir, "minicif_summary.csv"), index=False)
    crystal_system_summary.to_csv(
        os.path.join(out_dir, "minicif_crystal_system_summary.csv"),
        index=False,
    )
    metadata = dict(report_metadata or {})
    metadata.update({
        "checkpoint_files": [os.path.abspath(path) for path in paths],
        "checkpoint_splits": [
            os.path.splitext(os.path.basename(path))[0]
            for path in paths
        ],
        "prompt_modes": (
            sorted(results["prompt_mode"].dropna().unique().tolist())
            if "prompt_mode" in results
            else []
        ),
        "refinement_output": (
            os.path.abspath(refinement_output)
            if refinement_output
            else None
        ),
        "refinement_record_count": refinement_count,
        "summary": summary.to_dict(orient="records"),
    })
    with open(
        os.path.join(out_dir, "minicif_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2)

    if checkpoint is not None:
        plot_learning_curves(checkpoint, out_dir)
    plot_metric_summary(summary, out_dir)
    plot_refinement_metric_comparison(summary, out_dir)
    plot_rwp_distribution(results, out_dir)
    plot_best_rwp_cdf(results, out_dir)
    plot_crystal_system_metrics(crystal_system_summary, out_dir)
    plot_crystal_system_rwp(crystal_system_summary, out_dir)
    plot_crystal_system_rwp_distribution(results, out_dir)
    plot_rwp_vs_rmsd(results, out_dir)
    print(summary.to_string(index=False))
    print(f"Wrote combined minicif report to {out_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate minicif learning-curve and validation/test evaluation reports.")
    parser.add_argument("--checkpoint", default="", help="Path to ckpt.pt")
    parser.add_argument("--dataset-dir", default="", help="Dataset root containing serialized/{val,test}.h5; defaults to checkpoint config")
    parser.add_argument("--out-dir", default="", help="Output report directory; defaults to CHECKPOINT_DIR/minicif_report")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Rebuild combined outputs from checkpoints already in --out-dir",
    )
    parser.add_argument(
        "--restart-splits",
        action="store_true",
        help="Discard checkpoints for the requested --splits before evaluation",
    )
    parser.add_argument("--max-items", type=int, default=0, help="Limit items per split; 0 means all")
    parser.add_argument("--num-reps", type=int, default=4, help="Generated candidates per dataset item")
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--prompt-mode",
        choices=[
            "pxrd",
            "pxrd-elements",
            "pxrd-elements-cs",
            "pxrd-elements-cs-sg",
            "pxrd-stoichiometry",
            "pxrd-stoichiometry-cs",
            "pxrd-stoichiometry-cs-sg",
            "start",
            "constituents",
            "formula",
            "formula-cs",
            "formula-cs-sg",
        ],
        default="pxrd",
        help=(
            "Known-field prompt mode. pxrd starts from the representation start token; pxrd-elements "
            "also fixes constituent elements; pxrd-stoichiometry also fixes reduced formula counts."
        ),
    )
    parser.add_argument(
        "--prompt-modes",
        nargs="+",
        choices=[
            "pxrd",
            "pxrd-elements",
            "pxrd-elements-cs",
            "pxrd-elements-cs-sg",
            "pxrd-stoichiometry",
            "pxrd-stoichiometry-cs",
            "pxrd-stoichiometry-cs-sg",
            "start",
            "constituents",
            "formula",
            "formula-cs",
            "formula-cs-sg",
        ],
        default=None,
        help="Evaluate several prompt modes in one run. Defaults to --prompt-mode.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-current", action="store_true", help="Use current_model instead of best_model_state")
    parser.add_argument("--rmsd-threshold", type=float, default=0.0, help="Optional positive RMSD threshold for match rate")
    parser.add_argument("--cfg-scale", type=float, default=None, help="Classifier-free guidance scale (>1 strengthens PXRD conditioning; requires condition_dropout_prob>0 at train time)")
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Refine every valid generated candidate with BraggCalculator",
    )
    parser.add_argument(
        "--refinement-config",
        default="",
        help="YAML configuration for RefinementPolicy and optional species assignment",
    )
    parser.add_argument(
        "--refinement-domain",
        choices=["q", "two_theta"],
        default=None,
        help="Observed-pattern coordinate domain; overrides the refinement YAML",
    )
    parser.add_argument(
        "--refinement-radiation",
        choices=["xray", "neutron"],
        default=None,
        help="Radiation model; overrides the refinement YAML",
    )
    parser.add_argument(
        "--refinement-wavelength",
        type=float,
        default=None,
        help="Refinement wavelength in angstrom; defaults to --wavelength",
    )
    parser.add_argument(
        "--refinement-device",
        default=None,
        help="Torch device for refinement; defaults to --device",
    )
    parser.add_argument(
        "--refinement-policy",
        choices=["quick", "cautious", "robust"],
        default=None,
        help="RefinementPolicy preset; overrides the refinement YAML",
    )
    parser.add_argument(
        "--refine-best",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--refine-topk", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--refine-max-nfev", type=int, default=30, help=argparse.SUPPRESS)
    parser.add_argument(
        "--plot-examples",
        type=int,
        default=7,
        help=(
            "Maximum PXRD-plus-structure examples per split and prompt mode, "
            "stratified across reference crystal systems; 0 disables figures"
        ),
    )
    parser.add_argument(
        "--figure-supercell",
        type=int,
        default=1,
        help="Supercell repeat count for example structure panels",
    )
    parser.add_argument("--qmin", type=float, default=None)
    parser.add_argument("--qmax", type=float, default=None)
    parser.add_argument("--qstep", type=float, default=None)
    parser.add_argument("--clean-fwhm", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--wavelength", default="CuKa")
    parser.add_argument(
        "--artifact-config",
        default="",
        help="BraggCalculator artifact YAML used for deterministic robustness evaluation",
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    args.prompt_modes = args.prompt_modes or [args.prompt_mode]

    if args.combine_only:
        if not args.out_dir:
            parser.error("--out-dir is required with --combine-only")
        os.makedirs(args.out_dir, exist_ok=True)
        metadata = {}
        summary_path = os.path.join(args.out_dir, "minicif_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
            for key in (
                "checkpoint_files",
                "checkpoint_splits",
                "prompt_modes",
                "refinement_output",
                "refinement_record_count",
                "summary",
            ):
                metadata.pop(key, None)
        write_combined_report(args.out_dir, metadata)
        return
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --combine-only is used")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    checkpoint, model = load_checkpoint(args.checkpoint, device, use_best=not args.use_current)
    config = checkpoint_config(checkpoint)
    dataset_dir = args.dataset_dir or config.get("dataset")
    if not dataset_dir:
        raise ValueError("--dataset-dir is required when checkpoint config does not contain dataset")
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.checkpoint), "minicif_report")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer_name = checkpoint.get("model_args", {}).get("tokenizer", config.get("tokenizer", "minicif"))
    tokenizer, parse_fn, structure_fn, end_token = representation_api(tokenizer_name)
    matcher = StructureMatcher()
    xrd_kwargs = clean_xrd_kwargs(config, args)
    refinement_config = load_bragg_refinement_config(
        args.refinement_config,
        enabled=args.refine or args.refine_best,
        domain=args.refinement_domain,
        radiation=args.refinement_radiation,
        wavelength=args.refinement_wavelength,
        device=args.refinement_device,
        policy_name=args.refinement_policy,
        default_wavelength=XRDCalculator(
            wavelength=args.wavelength
        ).wavelength,
        default_device=args.device,
    )
    artifact_spec = (
        load_bragg_artifact_spec(args.artifact_config)
        if args.artifact_config
        else None
    )
    if artifact_spec is not None and artifact_spec.artifacts.seed is None:
        artifact_spec = replace(
            artifact_spec,
            artifacts=replace(artifact_spec.artifacts, seed=args.seed),
        )
    checkpoint_dir = os.path.join(out_dir, "evaluation_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    for split in args.splits:
        path = dataset_path(dataset_dir, split)
        split_checkpoint = EvaluationCheckpoint(
            os.path.join(checkpoint_dir, f"{split}.sqlite3"),
            evaluation_signature(
                args,
                args.checkpoint,
                path,
                tokenizer_name,
                xrd_kwargs,
                refinement_config,
            ),
            reset=args.restart_splits,
        )
        try:
            evaluate_split(
                split, path, model, tokenizer, parse_fn, structure_fn, end_token,
                matcher, xrd_kwargs, config, args, artifact_spec, out_dir,
                refinement_config, split_checkpoint,
            )
        finally:
            split_checkpoint.close()

    write_combined_report(
        out_dir,
        {
            "checkpoint": os.path.abspath(args.checkpoint),
            "dataset_dir": os.path.abspath(dataset_dir),
            "xrd_kwargs": xrd_kwargs,
            "artifact_config": (
                os.path.abspath(args.artifact_config)
                if args.artifact_config
                else None
            ),
            "refinement": refinement_config.to_dict(),
        },
        checkpoint=checkpoint,
    )


if __name__ == "__main__":
    main()
