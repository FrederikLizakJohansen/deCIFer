#!/usr/bin/env python
import argparse
import os
import re
import pickle
import random
from itertools import product
from pathlib import Path
import torch
import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

# Import necessary modules from your decifer package.
from decifer.config import AblationConfig, load_dataclass_config
from decifer.datasets import load_decifer_dataset, resolve_dataset_file
from decifer.evaluation import get_validity_flags
from decifer.io import create_run_layout
from decifer.utility import (
    pxrd_from_cif, 
    space_group_symbol_to_number,
    space_group_to_crystal_system,
)
from decifer.decifer_dataset import DeciferDataset
from decifer.generation import decode_and_fix_cif, extract_prompt, load_model_from_checkpoint
from decifer.tokenizer import Tokenizer

# Set up tokenizer constants.
tokenizer = Tokenizer()
PADDING_ID = tokenizer.padding_id
START_ID = tokenizer.token_to_id["data_"]
DECODE = tokenizer.decode


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run experiment and save results from config YAML.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file.")
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, default=None)
    parser.add_argument("--dataset-split", dest="dataset_split", type=str, default=None)
    parser.add_argument("--model-path", dest="model_path", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--n-repeats", dest="n_repeats", type=int, default=None)
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", dest="top_k", type=int, default=None)
    parser.add_argument("--add-composition", dest="add_composition", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--add-spacegroup", dest="add_spacegroup", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--crystal-system", dest="crystal_system", type=str, default=None)
    return parser


def parse_config(argv=None) -> AblationConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    overrides = vars(args).copy()
    config_path = overrides.pop("config")
    config = load_dataclass_config(AblationConfig, config_path=config_path, overrides=overrides)

    if not config.dataset_path:
        raise ValueError("The 'dataset_path' option is required and cannot be empty")
    if not config.model_path:
        raise ValueError("The 'model_path' option is required and cannot be empty")
    if not config.params_dict:
        raise ValueError("The 'params_dict' option is required and cannot be empty")

    return config


def resolve_run_dir(config: AblationConfig) -> str:
    output_path = Path(config.output)
    run_name = output_path.name
    for suffix in output_path.suffixes:
        if run_name.endswith(suffix):
            run_name = run_name[: -len(suffix)]
    run_name = run_name or "ablation_run"
    return str((output_path.parent / run_name).resolve())


def resolve_output_path(config: AblationConfig, layout) -> str:
    return os.path.join(layout.artifacts_dir, Path(config.output).name)

def experiment(
    cif_sample, 
    cif_tokens, 
    params_exp_dict, 
    model, 
    output_path, 
    config, 
    batch_size=1,
    n_repeats=1,
    cif_sample_other=None,
    default_params_dict = None,
    add_composition=True, 
    add_spacegroup=False,
    max_new_tokens=3076,
    temperature=1.0, 
    top_k=None,
):
    """
    Run an experiment by generating PXRD patterns and corresponding CIF outputs over a set of parameters.
    Uses batched generation (via generate_batched_reps) and a prompt extracted by extract_prompt,
    which is repeated to form a batch.
    
    Args:
        cif_sample (str): The baseline CIF string.
        cif_tokens: Token sequence for the prompt.
        params_exp_dict (dict): Dictionary mapping parameter names to lists of values (for pxrd_from_cif).
        model: The generative model with a generate_batched_reps() method.
        n_repeats (int): Number of repeats (batch size) per parameter combination.
        cif_sample_other (str, optional): Alternative CIF (for multi-phase experiments).
        add_composition (bool): Whether to add composition info to the prompt.
        add_spacegroup (bool): Whether to add spacegroup info to the prompt.
        max_new_tokens (int): Maximum new tokens to generate.
        temperature (float): Temperature parameter for generation.
        top_k (int): Top-K filtering parameter.
    
    Returns:
        dict: Dictionary with keys for each parameter combination and a list of experiment results.
              Each result contains the PXRD data (for input and generated CIFs), conditional vector,
              generated CIF, generated structure, reference structure, structure matching result,
              peak similarity, RWP, WD, and validity.
    """
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    params_tuples = [(key, val) for key in params_exp_dict.keys() for val in params_exp_dict[key]]
    #results = {key: {"param_values": {"value": None, "experiments": [], "best_experiment": None} for key in params_exp_dict.keys() for val in params_exp_dict[key]]}
    results = {key: {str(value): {"experiments": [], "best_experiment": None} for value in params_exp_dict[key]} for key in params_exp_dict.keys()}

    for i, (param_key, param_val) in enumerate(params_tuples):
        params_dict = {param_key: param_val}
        param_name = next(iter(params_dict.keys()))

        # Add default params
        if default_params_dict is not None:
            for key, val in default_params_dict.items():
                if key != param_key: # Only change if the key is not part of the experiment
                    params_dict[key] = val

        print(f"Processing parameter combination {i+1}/{len(params_tuples)}: {params_dict}")
        #results[param_name] = {"all": [], "best": None}
        
        # Generate the reference PXRD.
        cif_input = [cif_sample, cif_sample_other] if cif_sample_other is not None else cif_sample
        pxrd_ref = pxrd_from_cif(cif_input, debug=True, **params_dict)
        
        # Build the conditional vector from the continuous PXRD intensity.
        cond_vec = torch.from_numpy(pxrd_ref['iq']).unsqueeze(0).to(model.device)
        cond_vec = cond_vec.repeat(batch_size, 1)
        
        # Extract prompt (once) and replicate it for the batch.
        prompt_batch = extract_prompt(
            cif_tokens,
            device=model.device,
            add_composition=add_composition,
            add_spacegroup=add_spacegroup
        ).unsqueeze(0).repeat(batch_size, 1)

        best_result = None
        best_rwp = float("inf")
        for _ in range(n_repeats):
   
            # Batched generation.
            try:
                generated_batch = model.generate_batched_reps(
                    idx=prompt_batch,
                    max_new_tokens=max_new_tokens,
                    cond_vec=cond_vec,
                    start_indices_batch=[[0]] * batch_size,
                    temperature=temperature,
                    top_k=top_k,
                ).cpu().numpy()
            except Exception as e:
                print(f"Error during batched generation for parameter {param_name}: {e}")
                continue
            
            # Remove padding and decode each generated output.
                generated_batch = [ids[ids != PADDING_ID] for ids in generated_batch]
            
            for i, gen_ids in enumerate(generated_batch):
                try:
                    cif_string_gen = decode_and_fix_cif(gen_ids, tokenizer=tokenizer)
                    structure_gen = Structure.from_str(cif_string_gen, fmt="cif")
                    structure_ref = Structure.from_str(cif_sample, fmt="cif")
                    ref_param_val = params_dict[param_name]
                    if param_name == "q_pre_scale_abc":
                        lattice_matrix = structure_ref.lattice.matrix.copy()
                        lattice_matrix[0] *= ref_param_val[0]
                        lattice_matrix[1] *= ref_param_val[1]
                        lattice_matrix[2] *= ref_param_val[2]
                        structure_ref = Structure(Lattice(lattice_matrix), structure_ref.species, structure_ref.frac_coords)
                    elif param_name == "q_pre_scale_uniform":
                        lattice_matrix = structure_ref.lattice.matrix.copy() * ref_param_val
                        structure_ref = Structure(Lattice(lattice_matrix), structure_ref.species, structure_ref.frac_coords)

                    ref_lattice_lens = structure_ref.lattice.abc
                    ref_lattice_angs = structure_ref.lattice.angles
                    gen_lattice_lens = structure_gen.lattice.abc
                    gen_lattice_angs = structure_gen.lattice.angles
                    structure_rmsd = matcher.get_rms_dist(structure_ref, structure_gen)
                    structure_match = True if structure_rmsd is not None else False
                except Exception as e:
                    continue
                
                # Compute peak sim and Rwp
                if param_name in ["q_shift", "q_post_scale", "q_pre_scale_abc", "q_pre_scale_uniform"]:
                    pxrd_ref_clean = pxrd_from_cif(cif_sample, debug=True, **params_dict)
                    if default_params_dict is not None:
                        pxrd_gen_clean = pxrd_from_cif(cif_string_gen, debug=True, **default_params_dict)
                    else:
                        pxrd_gen_clean = pxrd_from_cif(cif_string_gen, debug=True)
                    pxrd_gen_matched = pxrd_gen_clean
                    rwp = np.sqrt(np.sum((pxrd_ref['iq'] - pxrd_gen_clean['iq'])**2) / np.sum(pxrd_ref['iq']**2))
                else:
                    if default_params_dict is not None:
                        pxrd_ref_clean = pxrd_from_cif(cif_sample, debug=True, **default_params_dict)
                        pxrd_gen_clean = pxrd_from_cif(cif_string_gen, debug=True, **default_params_dict)
                    else:
                        pxrd_ref_clean = pxrd_from_cif(cif_sample, debug=True)
                        pxrd_gen_clean = pxrd_from_cif(cif_string_gen, debug=True)
                    pxrd_gen_matched = pxrd_from_cif(cif_string_gen, debug=True, **params_dict)
                    rwp = np.sqrt(np.sum((pxrd_ref_clean['iq'] - pxrd_gen_clean['iq'])**2) / np.sum(pxrd_ref_clean['iq']**2))
                
                peak_similarity = np.corrcoef(pxrd_ref['iq'], pxrd_gen_matched['iq'])[0, 1]

                # Validity checks.
                try:
                    validity_flags = get_validity_flags(cif_string_gen)
                    val = all(validity_flags.values())
                except:
                    validity_flags = None
                    val = False
                
                exp_result = {
                    "iteration": i,
                    "pxrd_ref": pxrd_ref,
                    "pxrd_ref_clean": pxrd_ref_clean,
                    "pxrd_gen_clean": pxrd_gen_clean,
                    "pxrd_gen_matched": pxrd_gen_matched,
                    "cond_vec": cond_vec[i],
                    "generated_cif": cif_string_gen,
                    "reference_cif": cif_sample,
                    "generated_structure": structure_gen,
                    "reference_structure": structure_ref,
                    "structure_match": structure_match,
                    "structure_rmsd": structure_rmsd,
                    "peak_similarity": peak_similarity,
                    "rwp": rwp,
                    "val": val,
                    "validity": validity_flags,
                    "reference_lattice_lengths": ref_lattice_lens,
                    "reference_lattice_angles": ref_lattice_angs,
                    "generated_lattice_lenghts": gen_lattice_lens,
                    "generated_lattice_angles": gen_lattice_angs,
                }
                results[param_name][str(param_val)]["experiments"].append(exp_result)

                if rwp < best_rwp:
                    best_rwp = rwp
                    best_result = exp_result
                    
        results[param_name][str(param_val)]["best_experiment"] = best_result
        print(f"  Completed generation for {batch_size} x {n_repeats} repeats.")

        # Save the configuration and results.
        print(f"  Saving results...", end="")
        with open(output_path, "wb") as f:
            pickle.dump({"config": config, "results": results}, f)
        print(f"  DONE.")

    return results

def main(argv=None):
    config = parse_config(argv)
    run_dir = resolve_run_dir(config)
    layout = create_run_layout(
        run_dir,
        "ablation",
        config,
        metadata={
            "dataset_path": os.path.abspath(config.dataset_path),
            "resolved_dataset_path": resolve_dataset_file(config.dataset_path, split=config.dataset_split),
            "dataset_split": config.dataset_split,
            "model_path": os.path.abspath(config.model_path),
        },
    )

    # Use the seed from the config, defaulting to 100 if not provided.
    seed = config.seed
    random.seed(seed)
    
    dataset_path = config.dataset_path
    dataset = load_decifer_dataset(
        dataset_path,
        ["cif_name", "cif_tokens", "xrd.q", "xrd.iq", "cif_string", "spacegroup"],
        split=config.dataset_split,
        dataset_cls=DeciferDataset,
    )

    # Start with the full dataset
    filtered_dataset = dataset

    # Step 1: Filter by crystal system (if specified)
    target_system_name = config.crystal_system
    system_name_to_number = {
        "triclinic": 1,
        "monoclinic": 2,
        "orthorhombic": 3,
        "tetragonal": 4,
        "trigonal": 5,
        "hexagonal": 6,
        "cubic": 7,
    }

    if target_system_name:
        target_system_name_lower = target_system_name.lower()
        if target_system_name_lower not in system_name_to_number:
            raise ValueError(f"Unknown crystal system: {target_system_name}")
        target_system_number = system_name_to_number[target_system_name_lower]
        filtered_dataset = [
            d for d in filtered_dataset
            if space_group_to_crystal_system(space_group_symbol_to_number(d['spacegroup'])) == target_system_number
        ]

    # Step 2: Filter by element content (symbol match + optional count)
    target_elements = config.target_elements  # e.g., ["Fe", "O"]
    element_match_mode = config.element_match_mode.lower()  # "exact" or "contains"
    element_count = config.element_count  # e.g., 2 for binaries

    if target_elements or element_count is not None:
        target_elements_set = set(target_elements) if target_elements else None
        element_pattern = re.compile(r"_chemical_formula_sum\s+['\"]?([\w\d\s\(\)\.\-]+)['\"]?", re.IGNORECASE)

        def extract_elements_from_formula(formula_str):
            tokens = formula_str.split()
            found_elements = set()
            for token in tokens:
                match_elem = re.match(r"([A-Z][a-z]?)", token)
                if match_elem:
                    found_elements.add(match_elem.group(1))
            return found_elements

        def element_match_ok(cif_str):
            match = element_pattern.search(cif_str)
            if not match:
                return False
            found_elements = extract_elements_from_formula(match.group(1))

            # Apply element set matching
            if target_elements_set:
                if element_match_mode == "exact" and found_elements != target_elements_set:
                    return False
                elif element_match_mode == "contains" and not target_elements_set.issubset(found_elements):
                    return False

            # Apply element count constraint
            if element_count is not None and len(found_elements) != element_count:
                return False

            return True

        filtered_dataset = [
            d for d in filtered_dataset if element_match_ok(d['cif_string'])
        ]

    # Final check
    if not filtered_dataset:
        raise ValueError("No CIFs found matching the specified filters.")

    # Pick datapoints
    datapoint_1 = random.choice(filtered_dataset)
    datapoint_2 = random.choice(filtered_dataset)
    
    sample_cif = datapoint_1['cif_string']
    sample_tokens = datapoint_1['cif_tokens']
    
    # Extract configuration parameters.
    model_path = config.model_path
    params_dict = config.params_dict
    default_params_dict = config.default_params_dict
    batch_size = config.batch_size
    n_repeats = config.n_repeats
    max_new_tokens = config.max_new_tokens
    temperature = config.temperature
    top_k = config.top_k
    add_composition = config.add_composition
    add_spacegroup = config.add_spacegroup
    output_path = resolve_output_path(config, layout)
    
    sample_cif_2 = datapoint_2['cif_string'] if "phase_scales" in params_dict else None
    
    # Load the model.
    print(f"Loading model from {model_path} ...")
    model = load_model_from_checkpoint(model_path, device="cuda")
    
    # Run the experiment.
    print("Running experiment...")
    results = experiment(
        sample_cif,
        sample_tokens,
        params_dict,
        model,
        output_path=output_path,
        config=config,
        default_params_dict=default_params_dict,
        batch_size=batch_size,
        n_repeats=n_repeats,
        cif_sample_other=sample_cif_2,
        add_composition=add_composition,
        add_spacegroup=add_spacegroup,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    total_experiments = sum(
        len(value["experiments"])
        for param_results in results.values()
        for value in param_results.values()
    )
    layout.write_metadata(
        {
            "output_pickle": output_path,
            "sample_cif_name": datapoint_1["cif_name"],
            "secondary_cif_name": datapoint_2["cif_name"],
        }
    )
    layout.write_metrics(
        {
            "parameter_groups": len(results),
            "total_experiments": total_experiments,
            "has_multiphase_reference": sample_cif_2 is not None,
        }
    )

    print(f"Experiment complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
