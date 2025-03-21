#!/usr/bin/env python

import argparse
import os
import pickle
import warnings
from collections import defaultdict

import numpy as np
import torch

from pymatgen.analysis.structure_matcher import StructureMatcher

# decifer/Refactored imports
from bin.train import TrainConfig
from bin.evaluate import load_model_from_checkpoint, extract_prompt
from decifer.tokenizer import Tokenizer
from decifer.decifer_dataset import DeciferDataset
from decifer.utility import (
    extract_numeric_property,
    reinstate_symmetry_loop,
    replace_symmetry_loop_with_P1,
    extract_space_group_symbol,
    generate_continuous_xrd_from_cif,
    discrete_to_continuous_xrd,
    get_rmsd,
    space_group_symbol_to_number,
    space_group_to_crystal_system,
)

warnings.simplefilter("ignore")


def rwp(sample, gen):
    """
    Calculates the residual (un)weighted profile between a sample and a generated XRD pattern.
    """
    return np.sqrt(np.sum(np.square(sample - gen), axis=-1) / np.sum(np.square(sample), axis=-1))


def process_cif(
    cif_data,
    model,
    batch_size,
    num_reps,
    add_comp,
    add_spg,
    decode_fn,
    padding_id,
    qmin,
    qmax,
    qstep,
    fwhm_range,
    noise_range,
    matcher,
):
    """
    Generates CIFs from one sample (CIF) 'num_reps' times (in batches), returning Rwp & RMSD stats.
    """
    results = defaultdict(list)

    cif_name = cif_data["cif_name"]
    cif_sample = cif_data["cif_string"]
    cif_tokens = cif_data["cif_tokens"]
    xrd_disc_q = cif_data["xrd.q"]
    xrd_disc_iq = cif_data["xrd.iq"]
    spacegroup_symbol_sample = cif_data["spacegroup"]
    spacegroup_number_sample = space_group_symbol_to_number(spacegroup_symbol_sample)
    crystal_system_sample = space_group_to_crystal_system(spacegroup_number_sample)

    a_sample = extract_numeric_property(cif_sample, "_cell_length_a")
    b_sample = extract_numeric_property(cif_sample, "_cell_length_b")
    c_sample = extract_numeric_property(cif_sample, "_cell_length_c")
    alpha_sample = extract_numeric_property(cif_sample, "_cell_angle_alpha")
    beta_sample = extract_numeric_property(cif_sample, "_cell_angle_beta")
    gamma_sample = extract_numeric_property(cif_sample, "_cell_angle_gamma")

    # Prepare prompt and conditional vector
    prompt = extract_prompt(
        cif_tokens,
        model.device,
        add_composition=add_comp,
        add_spacegroup=add_spg
    ).unsqueeze(0)

    xrd = discrete_to_continuous_xrd(
        xrd_disc_q.unsqueeze(0),
        xrd_disc_iq.unsqueeze(0),
        qmin=qmin,
        qmax=qmax,
        qstep=qstep,
        fwhm_range=fwhm_range,
        noise_range=noise_range,
        mask_prob=None,
        intensity_scale_range=None,
    )
    iq = xrd["iq"]
    cond_vec = iq.to(model.device)

    # Generate CIF tokens in batch
    batch_prompt = prompt.repeat(batch_size, 1)
    cond_vec_batch = cond_vec.repeat(batch_size, 1)
    token_ids = []

    # We do multiple loops to collect all generations
    for i in range(num_reps):
        generated = model.generate_batched_reps(
            batch_prompt,
            max_new_tokens=3076,
            cond_vec=cond_vec_batch,
            start_indices_batch=[[0]] * batch_size,
        ).cpu().numpy()
        token_ids.extend(generated)
        print(f"Finished {i+1}/{num_reps} number of reps")

    # Remove padding tokens from each result
    token_ids = [ids[ids != padding_id] for ids in token_ids]

    # Post-process each generated CIF
    for tokens in token_ids:
        try:
            out_cif = decode_fn(list(tokens))
            out_cif = replace_symmetry_loop_with_P1(out_cif)
            spacegroup_symbol = extract_space_group_symbol(out_cif)
            if spacegroup_symbol != "P 1":
                out_cif = reinstate_symmetry_loop(out_cif, spacegroup_symbol)

            spacegroup_number_gen = space_group_symbol_to_number(spacegroup_symbol) 
            crystal_system_gen = space_group_to_crystal_system(spacegroup_number_gen)
            a_gen = extract_numeric_property(out_cif, "_cell_length_a")
            b_gen = extract_numeric_property(out_cif, "_cell_length_b")
            c_gen = extract_numeric_property(out_cif, "_cell_length_c")
            alpha_gen = extract_numeric_property(out_cif, "_cell_angle_alpha")
            beta_gen = extract_numeric_property(out_cif, "_cell_angle_beta")
            gamma_gen = extract_numeric_property(out_cif, "_cell_angle_gamma")

            # Generate XRD from generated CIF
            xrd_gen = generate_continuous_xrd_from_cif(
                out_cif,
                qmin=qmin,
                qmax=qmax,
                qstep=qstep,
                fwhm_range=fwhm_range,
                noise_range=noise_range,
                mask_prob=None,
                intensity_scale_range=None,
            )
            iq_gen = xrd_gen["iq"]

            # Calculate Rwp
            rwp_value = rwp(iq.cpu().numpy(), iq_gen)

            # StructureMatcher RMSD
            cif_rmsd = get_rmsd(cif_sample, out_cif, matcher)
        except Exception as e:
            print(e)
            continue

        # Store results
        results["cif_name"].append(cif_name)
        results["cif_string_gen"].append(out_cif)
        results["rwp"].append(rwp_value)
        results["rmsd"].append(cif_rmsd)
        results["spacegroup_gen"].append(spacegroup_number_gen)
        results["crystal_system_gen"].append(crystal_system_gen)
        results["a_gen"].append(a_gen)
        results["b_gen"].append(b_gen)
        results["c_gen"].append(c_gen)
        results["alpha_gen"].append(alpha_gen)
        results["beta_gen"].append(beta_gen)
        results["gamma_gen"].append(gamma_gen)

        results["cif_string_sample"].append(cif_sample)
        results["spacegroup_sample"].append(spacegroup_number_sample)
        results["crystal_system_sample"].append(crystal_system_sample)
        results["a_sample"].append(a_sample)
        results["b_sample"].append(b_sample)
        results["c_sample"].append(c_sample)
        results["alpha_sample"].append(alpha_sample)
        results["beta_sample"].append(beta_sample)
        results["gamma_sample"].append(gamma_sample)

    return results

def process_multiple_cifs(
    dataset_iter,
    model,
    num_cifs,
    batch_size=16,
    num_reps=1,
    add_comp=False,
    add_spg=False,
    decode_fn=None,
    padding_id=None,
    qmin=0.0,
    qmax=10.0,
    qstep=0.01,
    fwhm_range=(0.05, 0.05),
    noise_range=None,
    matcher=None,
):
    """
    Process Y CIFs from the dataset iterator, each generating N times in batches.
    """
    all_results = []
    for i, cif_data in enumerate(dataset_iter):
        if i >= num_cifs:
            break

        # Single-cif processing
        results = process_cif(
            cif_data,
            model=model,
            batch_size=batch_size,
            num_reps=num_reps,
            add_comp=add_comp,
            add_spg=add_spg,
            decode_fn=decode_fn,
            padding_id=padding_id,
            qmin=qmin,
            qmax=qmax,
            qstep=qstep,
            fwhm_range=fwhm_range,
            noise_range=noise_range,
            matcher=matcher,
        )
        all_results.append(results)
        print(f"Finished {i+1}/{num_cifs} number of CIFs")

    return all_results


def save_structures(results, folder):
    """
    Save each generated CIF into its own folder, named by the original CIF name.
    """
    os.makedirs(folder, exist_ok=True)
    for cif_results in results:
        # Each item in `cif_results` references a single original CIF
        cif_name = cif_results["cif_name"][0]
        subfolder = os.path.join(folder, cif_name)
        os.makedirs(subfolder, exist_ok=True)

        for i, cif_string in enumerate(cif_results["cif_string_gen"]):
            filename = os.path.join(subfolder, f"{cif_name}_{i}.cif")
            with open(filename, "w") as f:
                f.write(cif_string)


def main():
    parser = argparse.ArgumentParser(description="Generate CIFs from a pretrained model.")
    parser.add_argument("--num_cifs", type=int, required=True, help="Number of CIFs to process.")
    parser.add_argument("--num_reps", type=int, required=True, help="Number of times to generate each CIF.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--qmin", type=float, default=0.0)
    parser.add_argument("--qmax", type=float, default=10.0)
    parser.add_argument("--qstep", type=float, default=0.01)
    parser.add_argument("--add_comp", action='store_true')
    parser.add_argument("--add_spg", action='store_true')
    parser.add_argument("--fwhm", type=float, default=0.05)
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save CIFs/results.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model checkpoint.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the HDF5 dataset.")
    args = parser.parse_args()

    num_cifs = args.num_cifs
    num_reps = args.num_reps
    batch_size = args.batch_size
    qmin = args.qmin
    qmax = args.qmax
    qstep = args.qstep
    add_comp = args.add_comp
    add_spg = args.add_spg
    fwhm_range = (args.fwhm, args.fwhm)
    noise_range = (args.noise, args.noise) if args.noise is not None else None
    output_folder = args.output_folder
    model_path = args.model_path
    dataset_path = args.dataset_path

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from: {model_path}")
    model = load_model_from_checkpoint(model_path, device)
    model.eval()

    # Tokenizer references
    decode_fn = Tokenizer().decode
    padding_id = Tokenizer().padding_id

    print(f"Loading dataset from: {dataset_path}")
    dataset = DeciferDataset(dataset_path, ["cif_name", "cif_string", "cif_tokens", "xrd.q", "xrd.iq", "spacegroup"])
    dataset_iter = iter(dataset)

    print(f"Processing {num_cifs} CIF(s), generating each {num_reps} time(s)...")
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    results = process_multiple_cifs(
        dataset_iter,
        model=model,
        num_cifs=num_cifs,
        batch_size=batch_size,
        num_reps=num_reps,
        add_comp=add_comp,
        add_spg=add_spg,
        decode_fn=decode_fn,
        padding_id=padding_id,
        qmin=qmin,
        qmax=qmax,
        qstep=qstep,
        fwhm_range=fwhm_range,
        noise_range=noise_range,
        matcher=matcher,
    )

    os.makedirs(output_folder, exist_ok=True)
    pkl_path = os.path.join(output_folder, "results.pkl")
    print(f"Saving pickled results to {pkl_path}...")
    with open(pkl_path, "wb") as f:
         pickle.dump(results, f)

    print("Done!")

if __name__ == "__main__":
    main()
