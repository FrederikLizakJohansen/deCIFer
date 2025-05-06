#!/usr/bin/env python
import argparse
import yaml
import pickle
import random
from itertools import product
import torch
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

# Import necessary modules from your decifer package.
from decifer.utility import (
    pxrd_from_cif, 
    replace_symmetry_loop_with_P1, 
    extract_space_group_symbol, 
    reinstate_symmetry_loop, 
    is_formula_consistent, 
    is_space_group_consistent, 
    is_atom_site_multiplicity_consistent, 
    bond_length_reasonableness_score,
    space_group_symbol_to_number,
    space_group_to_crystal_system,
)
from decifer.decifer_dataset import DeciferDataset
from bin.evaluate import load_model_from_checkpoint, extract_prompt
from decifer.tokenizer import Tokenizer
from bin.train import TrainConfig

# Set up tokenizer constants.
tokenizer = Tokenizer()
PADDING_ID = tokenizer.padding_id
START_ID = tokenizer.token_to_id["data_"]
DECODE = tokenizer.decode

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
        combinatory (bool): Run combinations of parameters (grid-search), default False.
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
                    cif_string_gen = DECODE(gen_ids)
                    # Fix symmetry issues.
                    cif_string_gen = replace_symmetry_loop_with_P1(cif_string_gen)
                    spacegroup_symbol = extract_space_group_symbol(cif_string_gen)
                    if spacegroup_symbol != "P 1":
                        cif_string_gen = reinstate_symmetry_loop(cif_string_gen, spacegroup_symbol)
                    
                    structure_gen = Structure.from_str(cif_string_gen, fmt="cif")
                    structure_ref = Structure.from_str(cif_sample, fmt="cif")
                    # TODO ADD THE SHIFT TO THE STRUCTURE SOMEHOW
                    structure_match = matcher.fit(structure_ref, structure_gen)
                except:
                    continue
                
                
                # Compute peak sim and Rwp
                if param_name in ["q_shift", "q_scale"]:
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
                    form = is_formula_consistent(cif_string_gen)
                    sg = is_space_group_consistent(cif_string_gen)
                    mplt = is_atom_site_multiplicity_consistent(cif_string_gen)
                    bond = bond_length_reasonableness_score(cif_string_gen) >= 1.0
                    val = form and sg and mplt and bond
                except:
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
                    "peak_similarity": peak_similarity,
                    "rwp": rwp,
                    "val": val,
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

def main():
    parser = argparse.ArgumentParser(description="Run experiment and save results from config YAML.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file.")
    args = parser.parse_args()
    
    # Load the configuration.
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Use the seed from the YAML config, defaulting to 100 if not provided.
    seed = config.get("seed", 100)
    random.seed(seed)
    
    dataset_path = config["dataset_path"]
    dataset = DeciferDataset(dataset_path, ["cif_name", "cif_tokens", "xrd.q", "xrd.iq", "cif_string", "spacegroup"])
    
    # Optionally filter by a specific crystal system from the YAML config.
    # The user should insert the crystal system name (e.g., "triclinic", "monoclinic", etc.)
    target_system_name = config.get("crystal_system", None)
    # Mapping from crystal system name to its corresponding system number (1-7).
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
        filtered_dataset = [d for d in dataset if space_group_to_crystal_system(space_group_symbol_to_number(d['spacegroup'])) == target_system_number]
        if not filtered_dataset:
            raise ValueError(f"No datapoints found for crystal system: {target_system_name}")
        datapoint_1 = random.choice(filtered_dataset)
        datapoint_2 = random.choice(filtered_dataset)
    else:
        datapoint_1 = random.choice(dataset)
        datapoint_2 = random.choice(dataset)
    
    sample_cif = datapoint_1['cif_string']
    sample_tokens = datapoint_1['cif_tokens']
    
    # Extract configuration parameters.
    model_path = config["model_path"]
    params_dict = config["params_dict"]
    default_params_dict = config.get("default_params_dict", None)
    combinatory = config.get("combinatory", False)
    batch_size = config.get("batch_size", 1)
    n_repeats = config.get("n_repeats", 1)
    max_new_tokens = config.get("max_new_tokens", 3076)
    temperature = config.get("temperature", 1.0)
    top_k = config.get("top_k", None)
    add_composition = config.get("add_composition", False)
    add_spacegroup = config.get("add_spacegroup", False)
    output_path = config.get("output", "experiment_results.pkl")
    
    sample_cif_2 = datapoint_2['cif_string'] if "phase_scales" in params_dict else None
    
    # Load the model.
    print(f"Loading model from {model_path} ...")
    model = load_model_from_checkpoint(model_path, device="cuda")
    
    # Run the experiment.
    print("Running experiment...")
    experiment(
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
    
    print(f"Experiment complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
