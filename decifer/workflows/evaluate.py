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

# Third-party library imports
import torch
from tqdm.auto import tqdm
from pymatgen.io.cif import CifParser
from pymatgen.analysis.structure_matcher import StructureMatcher

# Conditional imports for backwards compatibility with older pymatgen versions
try:
    parser_from_string = CifParser.from_str
except AttributeError:
    parser_from_string = CifParser.from_string

from decifer.decifer_dataset import DeciferDataset
from decifer.config import EvaluateConfig, load_dataclass_config
from decifer.datasets import load_decifer_dataset, resolve_dataset_file
from decifer.evaluation import get_cif_statistics
from decifer.io import create_run_layout
from decifer.generation import (
    DECODE,
    PADDING_ID,
    extract_prompt,
    fix_generated_cif,
    generate_one,
    load_model_from_checkpoint,
)
from decifer.utility import (
    get_rmsd,
    is_sensible,
    discrete_to_continuous_xrd,
    generate_continuous_xrd_from_cif,
)


def evaluation_filename(structure_name: str, repetition_num: int) -> str:
    if "." in structure_name:
        structure_name = structure_name.split(".")[0]
    return f"{structure_name}_{repetition_num}.pkl.gz"


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Process and evaluate CIF files using multiprocessing.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")
    parser.add_argument("--model-ckpt", dest="model_ckpt", type=str, default=None, help="Path to the model ckpt file.")
    parser.add_argument("--root", type=str, default=None, help="Root directory path.")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=None, help="Number of worker processes.")
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, default=None, help="Path to the dataset HDF5 file.")
    parser.add_argument("--dataset-split", dest="dataset_split", type=str, default=None, help="Dataset split to use when dataset-path points to a dataset root.")
    parser.add_argument("--out-folder", dest="out_folder", type=str, default=None, help="Path to the output folder.")
    parser.add_argument("--debug-max", dest="debug_max", type=int, default=None, help="Maximum number of samples to process for debugging.")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None, help="Enable debug mode with additional output.")
    parser.add_argument("--add-composition", dest="add_composition", action=argparse.BooleanOptionalAction, default=None, help="Include composition in the prompt.")
    parser.add_argument("--add-spacegroup", dest="add_spacegroup", action=argparse.BooleanOptionalAction, default=None, help="Include spacegroup in the prompt.")
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=None, help="Maximum number of new tokens to generate.")
    parser.add_argument("--dataset-name", dest="dataset_name", type=str, default=None, help="Name of the dataset.")
    parser.add_argument("--model-name", dest="model_name", type=str, default=None, help="Name of the model.")
    parser.add_argument("--num-reps", dest="num_reps", type=int, default=None, help="Number of repetitions per sample.")
    parser.add_argument("--override", action=argparse.BooleanOptionalAction, default=None, help="Override the presence of existing files.")
    parser.add_argument("--condition", action=argparse.BooleanOptionalAction, default=None, help="Flag to condition the generations on XRD.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top-k", dest="top_k", type=int, default=None, help="Top-k sampling cutoff.")
    parser.add_argument("--add-noise", dest="add_noise", type=float, default=None, help="Add fixed noise to the XRD conditioning signal.")
    parser.add_argument("--add-broadening", dest="add_broadening", type=float, default=None, help="Add fixed peak broadening to the XRD conditioning signal.")
    parser.add_argument("--default_fwhm", type=float, default=None, help="Default FWHM for augmented XRD.")
    parser.add_argument("--clean_fwhm", type=float, default=None, help="FWHM for clean XRD generation.")
    parser.add_argument("--qmin", type=float, default=None)
    parser.add_argument("--qmax", type=float, default=None)
    parser.add_argument("--qstep", type=float, default=None)
    parser.add_argument("--wavelength", type=str, default=None)
    parser.add_argument("--eta", type=float, default=None)
    return parser


def parse_config(argv=None) -> EvaluateConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    overrides = vars(args).copy()
    config_path = overrides.pop("config")
    config = load_dataclass_config(EvaluateConfig, config_path=config_path, overrides=overrides)

    if not config.model_ckpt:
        raise ValueError("The 'model_ckpt' option is required and cannot be empty")
    if not config.dataset_path:
        raise ValueError("The 'dataset_path' option is required and cannot be empty")

    return config


def resolve_run_dir(config: EvaluateConfig) -> str:
    if config.out_folder:
        return config.out_folder
    base_dir = os.path.dirname(config.model_ckpt) or "."
    return os.path.join(base_dir, "runs", "evaluate", f"{config.dataset_name}__{config.model_name}")

def worker(input_queue, eval_files_dir, done_queue):
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
            'prompt_string': task.get('prompt_string'),
            'prompt_flags': task.get('prompt_flags'),
            'prompt_token_count': task.get('prompt_token_count'),
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
            'xrd_overlay_ready': False,
            'xrd_error': None,
            'status': status,
        }

        try:
            # Decode tokenized CIF structure into a string
            prompt_token_count = int(task.get('prompt_token_count') or 0)
            cif_token_gen_full = task['cif_token_gen']
            cif_token_gen_completion = cif_token_gen_full[prompt_token_count:]
            cif_string_gen_raw = DECODE(cif_token_gen_full)
            cif_string_completion_raw = DECODE(cif_token_gen_completion)
            cif_string_gen = fix_generated_cif(cif_string_gen_raw)
            
            status.append('syntax')
            evaluation_result_dict.update({
                'cif_string_gen_raw': cif_string_gen_raw,
                'cif_string_completion_raw': cif_string_completion_raw,
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
                xrd_clean_gen = generate_continuous_xrd_from_cif(
                    cif_string_gen,
                    structure_name = task['cif_name'],
                    debug = task['debug'],
                    **task['xrd_clean_dict'],
                )
                xrd_clean_sample = generate_continuous_xrd_from_cif(
                    task['cif_string_sample'],
                    structure_name = task['cif_name'],
                    debug = task['debug'],
                    **task['xrd_clean_dict'],
                )
                xrd_overlay_ready = bool(
                    xrd_clean_gen is not None
                    and xrd_clean_sample is not None
                    and xrd_clean_gen.get('q') is not None
                    and xrd_clean_gen.get('iq') is not None
                    and xrd_clean_sample.get('q') is not None
                    and xrd_clean_sample.get('iq') is not None
                )
                xrd_error = None if xrd_overlay_ready else "XRD generation returned no plottable arrays."
                evaluation_result_dict.update({
                    'xrd_clean_gen': xrd_clean_gen,
                    'xrd_clean_sample': xrd_clean_sample,
                    'xrd_overlay_ready': xrd_overlay_ready,
                    'xrd_error': xrd_error,
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
    output_filename = os.path.join(
        eval_files_dir, evaluation_filename(structure_name, repetition_num)
    )
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
    model: Any,
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
    dataset_split: Optional[str] = None,
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
        Tuple[int, int]: Number of requested `(sample, repetition)` evaluations and
        the number of tasks actually submitted.
    """
    # Load the dataset
    dataset = load_decifer_dataset(
        dataset_path,
        ["cif_name", "cif_tokens", "xrd.q", "xrd.iq", "cif_string", "spacegroup"],
        split=dataset_split,
        dataset_cls=DeciferDataset,
    )
    existing_eval_files = set(os.path.basename(f) for f in glob(os.path.join(eval_files_dir, "*.pkl.gz")))
    num_samples = len(dataset) if debug_max is None else min(len(dataset), debug_max)
    requested_tasks = num_samples * num_repetitions
    submitted_tasks = 0
    pbar = tqdm(total=requested_tasks, desc='Generating and parsing evaluation tasks...', leave=True)
    padding_id = PADDING_ID

    for i in range(num_samples):
        data = dataset[i]
        cif_name_sample = data['cif_name']
        cif_name_sample = cif_name_sample.split(".")[0]

        prompt = None if model is None else extract_prompt(
            data['cif_tokens'], model.device, add_composition, add_spacegroup
        ).unsqueeze(0)
        prompt_string = None if prompt is None else DECODE(prompt.squeeze(0).detach().cpu().tolist())
        prompt_token_count = 0 if prompt is None else int(prompt.size(1))

        xrd_input, cond_vec = None, None
        model_uses_condition = bool(getattr(getattr(model, "config", None), "condition", False))
        if prompt is not None and model_uses_condition:
            xrd_input = discrete_to_continuous_xrd(
                data['xrd.q'].unsqueeze(0), data['xrd.iq'].unsqueeze(0), **(xrd_augmentation_dict or {})
            )
            cond_vec = xrd_input['iq'].to(model.device)

        xrd_q_cont = xrd_input['q'].squeeze(0).numpy() if xrd_input else None
        xrd_iq_cont = xrd_input['iq'].squeeze(0).numpy() if xrd_input else None

        for rep_num in range(num_repetitions):
            output_basename = evaluation_filename(cif_name_sample, rep_num)
            if not override and output_basename in existing_eval_files:
                pbar.update(1)
                continue

            if prompt is not None:
                try:
                    cif_token_gen = generate_one(
                        model,
                        prompt,
                        max_new_tokens,
                        cond_vec=cond_vec,
                        start_indices_batch=[[0]],
                        temperature=temperature,
                        top_k=top_k,
                    )
                    cif_token_gen = cif_token_gen.numpy()
                except Exception as e:
                    print(f"Error in generating CIF for {cif_name_sample} rep {rep_num}: {e}")
                    pbar.update(1)
                    continue
            else:
                cif_token_gen = data['cif_tokens'][data['cif_tokens'] != padding_id].cpu().numpy()

            task = {
                'cif_name': cif_name_sample,
                'dataset_name': dataset_name,
                'model_name': model_name,
                'index': i,
                'rep': rep_num,
                'prompt_string': prompt_string,
                'prompt_token_count': prompt_token_count,
                'prompt_flags': {
                    'add_composition': add_composition,
                    'add_spacegroup': add_spacegroup,
                    'condition': model_uses_condition,
                },
                'xrd_q_discrete_sample': data['xrd.q'],
                'xrd_iq_discrete_sample': data['xrd.iq'],
                'xrd_q_continuous_sample': xrd_q_cont,
                'xrd_iq_continuous_sample': xrd_iq_cont,
                'xrd_clean_dict': xrd_clean_dict,
                'xrd_augmentation_dict': xrd_augmentation_dict,
                'cif_string_sample': data['cif_string'],
                'cif_token_sample': data['cif_tokens'],
                'cif_token_gen': cif_token_gen,
                'spacegroup_sample': data['spacegroup'],
                'debug': debug,
            }
            input_queue.put(task)
            submitted_tasks += 1
            pbar.update(1)

    pbar.close()
    for _ in range(num_workers):
        input_queue.put(None)

    return requested_tasks, submitted_tasks

def main(argv=None):
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
    config = parse_config(argv)

    resolved_dataset_path = resolve_dataset_file(config.dataset_path, split=config.dataset_split)

    if os.path.exists(config.model_ckpt):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_model_from_checkpoint(config.model_ckpt, device)
        model.eval()
    else:
        print(f"Checkpoint not found at {config.model_ckpt}")
        sys.exit(1)

    # Augmentation parameters
    if config.add_noise is not None:
        noise_range = (config.add_noise, config.add_noise)
    else:
        noise_range = None

    if config.add_broadening is not None:
        fwhm_range = (config.add_broadening, config.add_broadening)
    else:
        fwhm_range = (config.default_fwhm, config.default_fwhm)

    # Augmented XRD
    augmentation_dict = {
        'qmin': config.qmin,
        'qmax': config.qmax,
        'qstep': config.qstep,
        'wavelength': config.wavelength,
        'fwhm_range': fwhm_range,
        'eta_range': (config.eta, config.eta),
        'noise_range': noise_range,
        'intensity_scale_range': None,
        'mask_prob': None,
    }
    
    # Clean XRD
    clean_dict = {
        'qmin': config.qmin,
        'qmax': config.qmax,
        'qstep': config.qstep,
        'wavelength': config.wavelength,
        'fwhm_range': (config.clean_fwhm, config.clean_fwhm),
        'eta_range': (config.eta, config.eta),
        'noise_range': None,
        'intensity_scale_range': None,
        'mask_prob': None,
    }

    # Directory for evaluation
    run_dir = resolve_run_dir(config)
    layout = create_run_layout(
        run_dir,
        "evaluate",
        config,
        metadata={
            "dataset_name": config.dataset_name,
            "model_name": config.model_name,
            "dataset_path": os.path.abspath(config.dataset_path),
            "resolved_dataset_path": resolved_dataset_path,
            "dataset_split": config.dataset_split,
            "model_ckpt": os.path.abspath(config.model_ckpt),
        },
    )

    # Set up multiprocessing queue and directories for evaluation files
    input_queue = mp.Queue()
    done_queue = mp.Queue()
    eval_files_dir = os.path.join(layout.predictions_dir, "eval_files", config.dataset_name)
    os.makedirs(eval_files_dir, exist_ok=True)
    layout.write_metadata({"eval_files_dir": eval_files_dir})

    # Start worker processes for processing
    processes = [
        mp.Process(target=worker, args=(input_queue, eval_files_dir, done_queue))
        for _ in range(config.num_workers)
    ]
    
    for process in processes:
        process.start()

    # Start processing the dataset
    num_requested, num_send = process_dataset(
        dataset_path=config.dataset_path,
        model=model,
        input_queue=input_queue,
        eval_files_dir=eval_files_dir,
        num_workers=config.num_workers,
        override=config.override,
        debug_max=config.debug_max,
        debug=config.debug,
        add_composition=config.add_composition,
        add_spacegroup=config.add_spacegroup,
        max_new_tokens=config.max_new_tokens,
        dataset_name=config.dataset_name,
        model_name=config.model_name,
        num_repetitions=config.num_reps,
        xrd_augmentation_dict=augmentation_dict,
        xrd_clean_dict=clean_dict,
        temperature=config.temperature,
        top_k=config.top_k,
        dataset_split=config.dataset_split,
    )

    if num_send > 0:
        # Create a new progress bar for task completion
        pbar = tqdm(total=num_send, desc='Evaluating...', leave=True)
        # Monitor the done_queue and update the progress bar
        completed_tasks = 0
        while completed_tasks < num_send:
            try:
                # Wait for a task completion signal
                done_queue.get(timeout=1)
                # Update the progress bar
                pbar.update(1)
                completed_tasks += 1
            except Empty:
                pass

        pbar.close()

    layout.write_metrics(
        {
            "requested_tasks": num_requested,
            "submitted_tasks": num_send,
            "completed_tasks": num_send,
            "skipped_tasks": num_requested - num_send,
        }
    )

    # Join worker processes after processing is complete
    for process in processes:
        process.join()

if __name__ == '__main__':
    main()
