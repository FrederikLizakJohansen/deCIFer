#!/usr/bin/env python3

# Standard library imports
import os
import sys
import argparse
import multiprocessing as mp
from queue import Empty
from queue import Empty
from glob import glob
import pickle
import gzip
from typing import Any, Dict, Optional, Tuple
from warnings import warn

# Third-party library imports
import torch
import numpy as np
from tqdm.auto import tqdm
from pymatgen.io.cif import CifParser
from pymatgen.analysis.structure_matcher import StructureMatcher

# Conditional imports for backwards compatibility with older pymatgen versions
try:
    parser_from_string = CifParser.from_str
except AttributeError:
    parser_from_string = CifParser.from_string

from decifer.decifer_model import Decifer, DeciferConfig
from decifer.decifer_dataset import DeciferDataset
from decifer.tokenizer import Tokenizer
from decifer.utility import (
    get_rmsd,
    replace_symmetry_loop_with_P1,
    extract_space_group_symbol,
    reinstate_symmetry_loop,
    is_sensible,
    extract_numeric_property,
    get_unit_cell_volume,
    extract_volume,
    is_space_group_consistent,
    is_atom_site_multiplicity_consistent,
    is_formula_consistent,
    bond_length_reasonableness_score,
    extract_species,
    discrete_to_continuous_xrd,
    generate_continuous_xrd_from_cif,
)
from bin.train import TrainConfig

# Tokenizer, get start, padding and newline IDs
TOKENIZER = Tokenizer()
VOCAB_SIZE = TOKENIZER.vocab_size
START_ID = TOKENIZER.token_to_id["data_"]
PADDING_ID = TOKENIZER.padding_id
NEWLINE_ID = TOKENIZER.token_to_id["\n"]
SPACEGROUP_ID = TOKENIZER.token_to_id["_symmetry_space_group_name_H-M"]
DECODE = TOKENIZER.decode

def extract_prompt(sequence, device, add_composition=True, add_spacegroup=False):

    # Find "data_" and slice
    try:
        end_prompt_index = np.argwhere(sequence == START_ID)[0][0] + 1
    except IndexError:
        raise ValueError(f"'data_' id: {START_ID} not found in sequence", DECODE(sequence))

    # Add composition (and spacegroup)
    if add_composition:
        end_prompt_index += np.argwhere(sequence[end_prompt_index:] == NEWLINE_ID)[0][0]

        if add_spacegroup:
            end_prompt_index += np.argwhere(sequence[end_prompt_index:] == SPACEGROUP_ID)[0][0]
            end_prompt_index += np.argwhere(sequence[end_prompt_index:] == NEWLINE_ID)[0][0]
            
        end_prompt_index += 1
    
    prompt_ids = torch.tensor(sequence[:end_prompt_index].long()).to(device=device)

    return prompt_ids

def extract_prompt_batch(sequences, device, add_composition=True, add_spacegroup=False):
    """
    Extract prompt sequences from a batch of sequences and apply front padding (left-padding).
    """

    prompts = []
    prompt_lengths = []
    max_len = 0

    # Loop through each sequence in the batch
    for sequence in sequences:
        # Find "data_" and slice
        try:
            end_prompt_index = np.argwhere(sequence == START_ID)[0][0] + 1
        except IndexError:
            raise ValueError(f"'data_' id: {START_ID} not found in sequence", DECODE(sequence))

        # Handle composition and spacegroup additions
        if add_composition:
            end_prompt_index += np.argwhere(sequence[end_prompt_index:] == NEWLINE_ID)[0][0]
            if add_spacegroup:
                end_prompt_index += np.argwhere(sequence[end_prompt_index:] == SPACEGROUP_ID)[0][0]
                end_prompt_index += np.argwhere(sequence[end_prompt_index:] == NEWLINE_ID)[0][0]

        # Extract the prompt for this sequence
        prompt_ids = torch.tensor(sequence[:end_prompt_index + 1].long()).to(device)
        prompts.append(prompt_ids)
        prompt_lengths.append(len(prompt_ids))
        max_len = max(max_len, len(prompt_ids))

    # Apply left-padding: pad from the front to ensure all prompts are the same length
    padded_prompts = torch.full((len(sequences), max_len), PADDING_ID, dtype=torch.long).to(device)
    for i, prompt in enumerate(prompts):
        prompt_len = len(prompt)
        # Left-pad by inserting the prompt at the end of the padded sequence
        padded_prompts[i, -prompt_len:] = prompt

    return padded_prompts, prompt_lengths

# Function to load model from a checkpoint
def load_model_from_checkpoint(ckpt_path, device):
    
    # Checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)  # Load checkpoint
    state_dict = checkpoint.get("best_model_state", checkpoint.get("best_model"))
    
    model_args = checkpoint["model_args"]

    # Map renamed keys
    renamed_keys = {
        'cond_size': 'condition_size',
        'condition_with_mlp_emb': 'condition',
    }
    for old_key, new_key in renamed_keys.items():
        if old_key in model_args:
            model_args['use_old_model_format'] = True
            warn(
                f"'{old_key}' is deprecated and has been renamed to '{new_key}'. "
                "Please update your checkpoint or configuration files.",
                DeprecationWarning,
                stacklevel=2
            )
            model_args[new_key] = model_args.pop(old_key)
    
    # Remove unused keys
    removed_keys = [
        'use_lora',
        'lora_rank',
        'condition_with_cl_emb',
        'cl_model_ckpt',
        'freeze_condition_embedding',
    ]
    for removed_key in removed_keys:
        if removed_key in model_args:
            warn(
                f"'{removed_key}' is no longer used and will be ignored. "
                "Consider removing it from your checkpoint or configuration files.",
                DeprecationWarning,
                stacklevel=2
            )
            model_args.pop(removed_key)

    
    # Load the model and checkpoint
    model_config = DeciferConfig(**model_args)
    model = Decifer(model_config).to(device)
    model.device = device
    
    # Fix the keys of the state dict per CrystaLLM
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    #model.load_state_dict(state_dict)  # Load modified state_dict into the model
    model.load_state_dict(state_dict)
    return model

def safe_extract_boolean(extract_function, *args):
    try:
        result = extract_function(*args)
        return result
    except Exception:
        return False

def safe_extract(extract_function, *args):
    try:
        result = extract_function(*args)
        return result
    except Exception:
        return None

def get_cif_statistics(cif_string_gen, evaluation_result_dict):
    # Define the dictionary to hold the CIF statistics and validity checks
    stat_dict = {
        'validity': {
            'formula': False,
            'site_multiplicity': False,
            'bond_length': False,
            'spacegroup': False,
        },
        'cell_params': {
            'a': None,
            'b': None,
            'c': None,
            'alpha': None,
            'beta': None,
            'gamma': None,
            'implied_vol': None,
            'gen_vol': None,
        },
        'spacegroup': None,
        'species': None,
    }
    
    # Extract validity checks and update the validity dictionary
    stat_dict['validity']['formula'] = safe_extract_boolean(is_formula_consistent, cif_string_gen)
    stat_dict['validity']['site_multiplicity'] = safe_extract_boolean(is_atom_site_multiplicity_consistent, cif_string_gen)
    stat_dict['validity']['bond_length'] = safe_extract_boolean(lambda cif: bond_length_reasonableness_score(cif) >= 1.0, cif_string_gen)
    stat_dict['validity']['spacegroup'] = safe_extract_boolean(is_space_group_consistent, cif_string_gen)
    
    # Safely extract numeric properties of the unit cell
    a = safe_extract(extract_numeric_property, cif_string_gen, "_cell_length_a")
    b = safe_extract(extract_numeric_property, cif_string_gen, "_cell_length_b")
    c = safe_extract(extract_numeric_property, cif_string_gen, "_cell_length_c")
    alpha = safe_extract(extract_numeric_property, cif_string_gen, "_cell_angle_alpha")
    beta = safe_extract(extract_numeric_property, cif_string_gen, "_cell_angle_beta")
    gamma = safe_extract(extract_numeric_property, cif_string_gen, "_cell_angle_gamma")
    
    # Safely compute volumes
    implied_vol = safe_extract(get_unit_cell_volume, a, b, c, alpha, beta, gamma)
    gen_vol = safe_extract(extract_volume, cif_string_gen)
    
    # Update the 'cell_params' dictionary with the extracted properties
    stat_dict['cell_params'].update({
        'a': a,
        'b': b,
        'c': c,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'implied_vol': implied_vol,
        'gen_vol': gen_vol
    })
    stat_dict['spacegroup'] = safe_extract(extract_space_group_symbol, cif_string_gen)
    stat_dict['species'] = safe_extract(extract_species, cif_string_gen)

    if evaluation_result_dict is not None:
        stat_dict.update(evaluation_result_dict)
    
    return stat_dict

def worker(input_queue, eval_files_dir, done_queue):
    
    # Initialize the tokenizer decoder function
    tokenizer = Tokenizer()
    decode = tokenizer.decode

    # Initialise pymatgen Matcher
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)

    while True:
        # Fetch task from the input queue
        task = input_queue.get()

        status = []

        # If a `None` task is received, terminate the worker
        if task is None:
            break

        status.append('task')

        evaluation_result_dict = {
            'cif_name': task['cif_name'],
            'dataset_name': task.get('dataset_name', 'N/A'),
            'model_name': task.get('model_name', 'N/A'),
            'index': task['index'],
            'rep': task['rep'],
            'xrd_clean_dict': task['xrd_clean_dict'],
            'xrd_augmentation_dict': task['xrd_augmentation_dict'],
            'cif_string_sample': task['cif_string_sample'],
            'cif_token_sample': task.get('cif_token_sample', None),
            'spacegroup_sample': task.get('spacegroup_sample', None),
            'xrd_q_discrete_sample': task['xrd_q_discrete_sample'],
            'xrd_iq_discrete_sample': task['xrd_iq_discrete_sample'],
            'xrd_q_continuous_sample': task['xrd_q_continuous_sample'],
            'xrd_iq_continuous_sample': task['xrd_iq_continuous_sample'],
            'seq_len_sample': len(task['cif_token_sample']),
            'seq_len_gen': len(task['cif_token_gen']),
            'status': status,
        }

        try:
            # Decode tokenized CIF structure into a string
            cif_string_gen = decode(task['cif_token_gen'])
            
            # Replace symmetry loop with primitive cell ("P1")
            cif_string_gen = replace_symmetry_loop_with_P1(cif_string_gen)

            # Extract space group symbol from the CIF string
            spacegroup_symbol = extract_space_group_symbol(cif_string_gen)

            # If the space group is not "P1", reinstate symmetry
            if spacegroup_symbol != "P 1":
                cif_string_gen = reinstate_symmetry_loop(cif_string_gen, spacegroup_symbol)
            
            status.append('syntax')
            evaluation_result_dict.update({
                'cif_string_gen': cif_string_gen,
                'status': status,
            })

            # Check if the CIF structure is sensible
            if is_sensible(cif_string_gen):

                status.append('sensible')
                evaluation_result_dict.update({'status': status})

                # Evaluate CIF validity and structure
                evaluation_result_dict = get_cif_statistics(cif_string_gen, evaluation_result_dict)

                # Evaluate matching structures by RMSD
                rmsd = get_rmsd(task['cif_string_sample'], cif_string_gen, matcher=matcher)
                evaluation_result_dict.update({'rmsd': rmsd})
            
                status.append('statistics')
                evaluation_result_dict.update({'status': status})

                # Calculate clean xrd
                evaluation_result_dict.update({
                    'xrd_clean_gen': generate_continuous_xrd_from_cif(
                        cif_string_gen,
                        structure_name = task['cif_name'],
                        debug = task['debug'],
                        **task['xrd_clean_dict'],
                    ),
                    'xrd_clean_sample': generate_continuous_xrd_from_cif(
                        task['cif_string_sample'],
                        structure_name = task['cif_name'],
                        debug = task['debug'],
                        **task['xrd_clean_dict'],
                    )
                })

                status.append('success')
                evaluation_result_dict.update({'status': status})

            save_evaluation(evaluation_result_dict, task['cif_name'], task['rep'], eval_files_dir)

        except Exception as e:
            # In case of error, save error information
            status.append('error')
            evaluation_result_dict.update({'status': status, 'error_msg': str(e)})
            save_evaluation(evaluation_result_dict, task['cif_name'], task['rep'], eval_files_dir)
        finally:
            # Signal task completion
            done_queue.put(1)

def save_evaluation(
    eval_result: dict,
    structure_name: str,
    repetition_num: int, 
    eval_files_dir: str
) -> None:
    """
    Save the evaluation result to a compressed pickle file.

    Args:
        eval_result (dict): Evaluation result to save.
        structure_name (str): Name of the structure being evaluated.
        repetition_num (int): Repetition number for the evaluation.
        eval_files_dir (str): Directory to save the evaluation file.
    """
    if "." in structure_name:
        structure_name = structure_name.split(".")[0]
    output_filename = os.path.join(eval_files_dir, f"{structure_name}_{repetition_num}.pkl.gz")
    temp_filename = output_filename + '.tmp'

    try:
        with gzip.open(temp_filename, 'wb') as temp_file:
            pickle.dump(eval_result, temp_file)
        os.rename(temp_filename, output_filename)
    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        raise IOError(f"Failed to save evaluation for {structure_name} (rep {repetition_num}): {e}")

def process_dataset(
    dataset_path: str,
    dataset_name: str,
    model: Decifer,
    model_name: str = "",
    input_queue: Any = None,
    eval_files_dir: str = "./eval_files",
    num_workers: int = 4,
    override: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
    num_repetitions: int = 1,
    add_composition: bool = False,
    add_spacegroup: bool = False,
    xrd_augmentation_dict: Optional[Dict] = None,
    xrd_clean_dict: Optional[Dict] = None,
    max_new_tokens: int = 256,
    debug_max: Optional[int] = None,
    debug: bool = False,
) -> Tuple[int, int]:
    """
    Processes a dataset for evaluation by generating tasks for model inference.

    Args:
        dataset_path (str): Path to the HDF5 dataset file.
        dataset_name (str): Name of the dataset.
        model (Decifer): Model object for inference.
        model_name (str): Name of the model. Defaults to "".
        input_queue (Any): Multiprocessing queue for task communication. Defaults to None.
        eval_files_dir (str): Directory to store evaluation files. Defaults to "./eval_files".
        num_workers (int): Number of worker processes. Defaults to 4.
        override (bool): Whether to override existing evaluation files. Defaults to False.
        temperature (float): Temperature for sampling during generation. Defaults to 1.0.
        top_k (int): Top-K sampling parameter. Defaults to 50.
        num_repetitions (int): Number of repetitions per dataset sample. Defaults to 1.
        add_composition (bool): Whether to include composition information in prompts. Defaults to False.
        add_spacegroup (bool): Whether to include spacegroup information in prompts. Defaults to False.
        xrd_augmentation_dict (Optional[Dict]): XRD augmentation parameters. Defaults to None.
        xrd_clean_dict (Optional[Dict]): XRD cleaning parameters. Defaults to None.
        max_new_tokens (int): Maximum number of tokens to generate. Defaults to 256.
        debug_max (Optional[int]): Debug mode limit for maximum samples to process. Defaults to None.
        debug (bool): Enable debug mode. Defaults to False.

    Returns:
        Tuple[int, int]: Number of generations processed and number of tasks sent.
    """
    # Load the dataset
    dataset = DeciferDataset(dataset_path, ["cif_name", "cif_tokens", "xrd.q", "xrd.iq", "cif_string", "spacegroup"])
    existing_eval_files = set(os.path.basename(f) for f in glob(os.path.join(eval_files_dir, "*.pkl.gz")))
    num_generations = len(dataset) * num_repetitions - len(existing_eval_files) if debug_max is None else min(len(dataset) * num_repetitions, debug_max)
    num_send = num_generations
    pbar = tqdm(total=num_generations, desc='Generating and parsing evaluation tasks...', leave=True)
    padding_id = Tokenizer().padding_id

    for i, data in enumerate(iter(dataset)):
        if i >= num_generations:
            break

        cif_name_sample = data['cif_name']
        cif_name_sample = cif_name_sample.split(".")[0]
        if not override and any(f.startswith(cif_name_sample) for f in existing_eval_files):
            pbar.update(1)
            num_send -= 1
            continue

        prompt = None if model is None else extract_prompt(
            data['cif_tokens'], model.device, add_composition, add_spacegroup
        ).unsqueeze(0)

        xrd_input, cond_vec, cif_token_gen = None, None, None
        if prompt is not None:
            xrd_input = discrete_to_continuous_xrd(
                data['xrd.q'].unsqueeze(0), data['xrd.iq'].unsqueeze(0), **(xrd_augmentation_dict or {})
            )
            cond_vec = xrd_input['iq'].to(model.device)
            try:
                cif_token_gen = model.generate_batched_reps(
                    prompt, max_new_tokens, cond_vec, [[0]], temperature, top_k
                ).cpu().numpy()
            except:
                print(f"Error in generating CIF for {cif_name_sample}")
                pbar.update(1)
                num_send -= 1
                continue
            cif_token_gen = [ids[ids != padding_id] for ids in cif_token_gen]
        else:
            cif_token_gen = [data['cif_tokens'][data['cif_tokens'] != padding_id].cpu().numpy()]

        xrd_q_cont = xrd_input['q'].squeeze(0).numpy() if xrd_input else None
        xrd_iq_cont = xrd_input['iq'].squeeze(0).numpy() if xrd_input else None

        for rep_num in range(num_repetitions):
            task = {
                'cif_name': cif_name_sample,
                'dataset_name': dataset_name,
                'model_name': model_name,
                'index': i,
                'rep': rep_num,
                'xrd_q_discrete_sample': data['xrd.q'],
                'xrd_iq_discrete_sample': data['xrd.iq'],
                'xrd_q_continuous_sample': xrd_q_cont,
                'xrd_iq_continuous_sample': xrd_iq_cont,
                'xrd_clean_dict': xrd_clean_dict,
                'xrd_augmentation_dict': xrd_augmentation_dict,
                'cif_string_sample': data['cif_string'],
                'cif_token_sample': data['cif_tokens'],
                'cif_token_gen': cif_token_gen[rep_num],
                'spacegroup_sample': data['spacegroup'],
                'debug': debug,
            }
            input_queue.put(task)
        pbar.update(1)

    pbar.close()
    for _ in range(num_workers):
        input_queue.put(None)

    return num_generations, num_send

def main():
    """
    Main function to process and evaluate CIF files using multiprocessing.

    Command-line Arguments:
        --config-path (str): Path to the YAML configuration file.
        --root (str): Root directory path (default: current directory).
        --num-workers (int): Number of worker processes to use (default: number of CPU cores minus one).
        --dataset-path (str): Path to the dataset HDF5 file.
        --out-folder (str): Directory where output files will be saved.
        --debug-max (int): Maximum number of samples to process in debug mode.
        --debug (bool): Enable debug mode with additional output.
        --no-model (bool): Disable model usage and evaluate from dataset only.
        --add-composition (bool): Include composition information in the prompt.
        --add-spacegroup (bool): Include spacegroup information in the prompt.
        --max-new-tokens (int): Maximum number of new tokens to generate for CIF structures.
        --dataset-name (str): Name of the dataset for saving evaluation results.
        --model-name (str): Name of the model for saving evaluation results.
        --num-reps (int): Number of repetitions per sample for CIF generation.

    Returns:
        None: The function manages dataset processing, task distribution, and evaluation.
    """
    parser = argparse.ArgumentParser(description='Process and evaluate CIF files using multiprocessing.')

    # Argument parsing for required and optional arguments
    parser.add_argument('--model-ckpt', type=str, help='Path to the model ckpt file.')
    parser.add_argument('--root', type=str, default='./', help='Root directory path.')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1, help='Number of worker processes.')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset HDF5 file.')
    parser.add_argument('--out-folder', type=str, default=None, help='Path to the output folder.')
    parser.add_argument('--debug-max', type=int, default=None, help='Maximum number of samples to process for debugging.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional output.')
    parser.add_argument('--add-composition', action='store_true', help='Include composition in the prompt.')
    parser.add_argument('--add-spacegroup', action='store_true', help='Include spacegroup in the prompt.')
    parser.add_argument('--max-new-tokens', type=int, default=1000, help='Maximum number of new tokens to generate.')
    parser.add_argument('--dataset-name', type=str, default='default_dataset', help='Name of the dataset.')
    parser.add_argument('--model-name', type=str, default='default_model', help='Name of the model.')
    parser.add_argument('--num-reps', type=int, default=1, help='Number of repetitions per sample.')
    parser.add_argument('--override', action='store_true', help='Overrides the presence of existing files, effectively generating everything from scratch.')
    parser.add_argument('--condition', action='store_true', help='Flag to condition the generations on XRD.')
    parser.add_argument('--temperature', type=float, default=1.0, help='')
    parser.add_argument('--top-k', type=int, default=None, help='')
    parser.add_argument('--add-noise', type=float, default=None, help='')
    parser.add_argument('--add-broadening', type=float, default=None, help='')
    parser.add_argument('--default_fwhm', type=float, default=0.05, help='')
    parser.add_argument('--clean_fwhm', type=float, default=0.05, help='')
    parser.add_argument('--qmin', type=float, default=0.0)
    parser.add_argument('--qmax', type=float, default=10.0)
    parser.add_argument('--qstep', type=float, default=0.01)
    parser.add_argument('--wavelength', type=str, default='CuKa')
    parser.add_argument('--eta', type=float, default=0.5)

    # Parse command-line arguments
    args = parser.parse_args()

    if os.path.exists(args.model_ckpt):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model_from_checkpoint(args.model_ckpt, device)
        model.eval()
    else:
        print(f"Checkpoint not found at {args.model_ckpt}")
        sys.exit(1)

    # Augmentation parameters
    if args.add_noise is not None:
        noise_range = (args.add_noise, args.add_noise)
    else:
        noise_range = None

    if args.add_broadening is not None:
        fwhm_range = (args.add_broadening, args.add_broadening)
    else:
        fwhm_range = (args.default_fwhm, args.default_fwhm)

    # Augmented XRD
    augmentation_dict = {
        'qmin': args.qmin,
        'qmax': args.qmax,
        'qstep': args.qstep,
        'wavelength': args.wavelength,
        'fwhm_range': fwhm_range,
        'eta_range': (args.eta, args.eta),
        'noise_range': noise_range,
        'intensity_scale_range': None,
        'mask_prob': None,
    }
    
    # Clean XRD
    clean_dict = {
        'qmin': args.qmin,
        'qmax': args.qmax,
        'qstep': args.qstep,
        'wavelength': args.wavelength,
        'fwhm_range': (args.clean_fwhm, args.clean_fwhm),
        'eta_range': (args.eta, args.eta),
        'noise_range': None,
        'intensity_scale_range': None,
        'mask_prob': None,
    }

    # Directory for evaluation
    if args.out_folder is not None:
        out_folder = args.out_folder
        os.makedirs(out_folder, exist_ok=True)
    else:
        out_folder = os.path.dirname(args.model_ckpt)

    # Set up multiprocessing queue and directories for evaluation files
    input_queue = mp.Queue()
    done_queue = mp.Queue()
    eval_files_dir = os.path.join(out_folder or os.path.dirname(args.dataset_path), "eval_files", args.dataset_name)
    os.makedirs(eval_files_dir, exist_ok=True)

    # Start worker processes for processing
    processes = [
        mp.Process(target=worker, args=(input_queue, eval_files_dir, done_queue))
        for _ in range(args.num_workers)
    ]
    
    for process in processes:
        process.start()

    # Start processing the dataset
    num_gens, num_send = process_dataset(
        dataset_path=args.dataset_path,
        model=model,
        input_queue=input_queue,
        eval_files_dir=eval_files_dir,
        num_workers=args.num_workers,
        debug_max=args.debug_max,
        debug=args.debug,
        add_composition=args.add_composition,
        add_spacegroup=args.add_spacegroup,
        max_new_tokens=args.max_new_tokens,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        num_repetitions=args.num_reps,
        xrd_augmentation_dict=augmentation_dict,
        xrd_clean_dict=clean_dict,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    if num_send > 0:
        # Create a new progress bar for task completion
        pbar = tqdm(total=num_gens, desc='Evaluating...', leave=True)
        # Monitor the done_queue and update the progress bar
        completed_tasks = 0
        while completed_tasks < num_gens:
            try:
                # Wait for a task completion signal
                done_queue.get(timeout=1)
                # Update the progress bar
                pbar.update(1)
                completed_tasks += 1
            except Empty:
                pass

        pbar.close()

    # Join worker processes after processing is complete
    for process in processes:
        process.join()

if __name__ == '__main__':
    main()
