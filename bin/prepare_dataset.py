#!/usr/bin/env python3

import os
import gc
import h5py
import gzip
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob

from pymatgen.io.cif import CifWriter, Structure, CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Composition
from typing import List, Optional

try:
    parser_from_string = CifParser.from_str
except:
    parser_from_string = CifParser.from_string

from sklearn.model_selection import train_test_split

import multiprocessing as mp
from multiprocessing import Pool, cpu_count, TimeoutError

import logging
from logging import handlers

from decifer.tokenizer import Tokenizer
from decifer.utility import (
    replace_symmetry_loop_with_P1,
    remove_cif_header,
    remove_oxidation_loop,
    format_occupancies,
    extract_formula_units,
    extract_formula_nonreduced,
    extract_space_group_symbol,
    replace_data_formula_with_nonreduced_formula,
    round_numbers,
    add_atomic_props_block,
)

import warnings
warnings.filterwarnings("ignore")

def init_worker(log_queue=None):
    """
    Initializes each worker process with a queue-based logger and loads the model if provided.
    This will be called when each worker starts.
    """

    # Initialize logger
    if log_queue:
        queue_handler = handlers.QueueHandler(log_queue)
        logger = logging.getLogger()
        logger.addHandler(queue_handler)
        logger.setLevel(logging.ERROR)
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.ERROR)

def log_listener(queue, log_dir):
    """
    Function excecuted by the log listener process.
    Receives messages from the queue and writes them to the log file.
    Rotates the log file based on size.
    """
    # Setup file handler (rotating)
    log_file = os.path.join("./" + log_dir, "error.log")
    handler = handlers.RotatingFileHandler(
        log_file, maxBytes=1024*1024, backupCount=5,
    )
    handler.setLevel(logging.ERROR)

    # Add a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Set up the log listener
    listener = handlers.QueueListener(queue, handler)
    listener.start()

    # return listener
    return listener

def save_metadata(metadata, data_dir):
    """
    Save metadata information (sizes, shapes, argument parameters, vocab size, etc.) into a centralized JSON file.

    Args:
        metadata (dict): Dictionary containing metadata information.
        data_dir (str): Directory where the metadata file should be saved.
    """
    metadata_file = os.path.join(data_dir, "metadata.json")
    
    # If the metadata file already exists, load the existing metadata and update it
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            existing_metadata = json.load(f)
        # Update existing metadata with new data
        existing_metadata.update(metadata)
        metadata = existing_metadata
    
    # Save updated metadata to the JSON file
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
def create_stratification_key(pmg_structure, group_size):
    """
    Create a stratification key from the spacegroup.

    Args:
        pmg_structure: pymatgen Structure object.
        group_size (int): Size of the spacegroup bin.

    Returns:
        str: Stratification key combining spacegroup in defined groups.
    """
    
    spacegroup_number = pmg_structure.get_space_group_info()[1]
    group_start = ((spacegroup_number - 1) // group_size) * group_size + 1
    group_end = group_start + group_size - 1

    return f"{group_start}-{group_end}"

def safe_pkl_gz(output, output_path):
    temp_path = output_path + '.tmp' # Ensuring that only fully written files are considered when collecting
    with gzip.open(temp_path, 'wb') as f:
        pickle.dump(output, f)
    os.rename(temp_path, output_path) # File is secure, renaming

def run_subtasks(
    root: str,
    worker_function,
    get_from: str,
    save_to: str,
    add_metadata: List[str] = [],
    task_kwargs_dict = {},
    announcement: Optional[str] = None,
    debug: bool = False,
    debug_max: Optional[int] = None,
    num_workers: int = cpu_count() - 1,
    from_gzip: bool = False,
):
    if announcement:
        print("-"*20)
        print(announcement)
        print("-"*20)

    # Locate pickles
    from_dir = os.path.join(root, get_from)
    pickles = glob(os.path.join(from_dir, '*.pkl.gz'))
    if worker_function.__name__ == "preprocess_worker":
        if not from_gzip:
            cifs = glob(os.path.join(from_dir, 'raw/*.cif'))
            assert len(cifs) > 0, f"Cannot locate any files in {from_dir}"
            paths = sorted(cifs)[:debug_max]
            assert len(paths) > 1, f"Flagging suspicious behaviour, only 1 or less CIFs present in {from_dir}: {paths}"
        else:
            assert len(pickles) == 1, f"from_gzip flag is raised, but more than one gzip file found in directory"
            with gzip.open(pickles[0], 'rb') as f:
                paths = pickle.load(f)[:debug_max]
                paths = sorted(paths, key=lambda x: x[0])
    else:
        paths = pickles

    # Make output folder
    to_dir = os.path.join(root, save_to)
    os.makedirs(to_dir, exist_ok=True)

    # Open metadata if specified
    if add_metadata:
        metadata_path = os.path.join(root, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        for key in add_metadata:
            try:
                task_kwargs_dict[key] = metadata[key]
            except NameError:
                print(f"Could not locate metadata with key: {key} in {metadata_path}")
                pass

    # Check for existing files
    existing_files = set(os.path.basename(f) for f in glob(os.path.join(to_dir, "*.pkl.gz")))

    # Collect tasks
    tasks = []
    pbar = tqdm(desc="Collecting tasks...", total=len(paths), leave=False)
    for path in paths:
        if isinstance(path, str):
            name = os.path.basename(path)
        else:
            name, _ = path
            name = name + '.pkl.gz'
        if name in existing_files:
            pbar.update(1)
            continue
        else:
            tasks.append(
                (path, task_kwargs_dict, debug, to_dir)
            )
            pbar.update(1)
    pbar.close()

    if not tasks:
        print(f"All tasks have already been executed for {save_to}, skipping...", end='\n')
    else:
        # Initialize logger
        log_queue = mp.Queue()
        listener = log_listener(log_queue, to_dir)

        # Parallel processing of CIF files using multiprocessing
        with Pool(processes=num_workers, initializer=init_worker, initargs=(log_queue,)) as pool:
            results_iterator = pool.imap_unordered(worker_function, tasks)

            for _ in tqdm(range(len(tasks)), total=len(tasks), desc="Executing tasks...", leave=False):
                try:
                    results_iterator.next(timeout=60)
                except TimeoutError:
                    print("Process timed out.")
                    continue

        # Stop log listener and flush
        listener.stop()
        logging.shutdown()

    n_files = len(paths)
    n_successful_files = len(glob(os.path.join(to_dir, '*.pkl.gz')))
    print(f"Reduction in dataset: {n_files} --> {n_successful_files} samples")

    additional_metadata = {save_to: task_kwargs_dict}
    save_metadata(additional_metadata, root)

    # Free up memory
    del tasks
    gc.collect()

def preprocess_worker(args):
    
    # Extract arguments
    obj, task_dict, debug, to_dir = args
    cif_name = "Unknown"

    try:
        # Make structure
        if isinstance(obj, str): # Path
            cif_name = os.path.basename(obj)
            try:
                structure = Structure.from_file(obj)
            except:
                raise Exception("Unexpected format found in preprocessing step of {obj} of type str")
        elif isinstance(obj, tuple): # cif string from pkl
            cif_name, cif_string = obj
            try:
                structure = parser_from_string(cif_string).get_structures()[0]
            except:
                raise Exception("Unexpected format found in preprocessing step of {obj} of type tuple")
        else:
            raise Exception("Unexpected type found in preprocessing step of {obj}")
        
        # Option for removing structures with occupancies below 1
        if not task_dict['include_occupancy_structures']:
            for site in structure:
                occ = list(site.species.as_dict().values())[0]
                if occ < 1:
                    raise Exception("Occupancy below 1.0 found")

        # Get stratification key
        strat_key = create_stratification_key(structure, task_dict['spacegroup_strat_group_size'])

        # Get raw content of CIF in string
        cif_string = CifWriter(struct=structure, symprec=0.1).__str__()

        # Extract formula units and remove if Z=0
        formula_units = extract_formula_units(cif_string)
        if formula_units == 0:
            raise Exception()

        # Remove oxidation state information
        cif_string = remove_oxidation_loop(cif_string)

        # Number precision rounding
        cif_string = round_numbers(cif_string, task_dict['num_decimal_places'])
        cif_string = format_occupancies(cif_string, task_dict['num_decimal_places'])

        # Add atomic props block
        cif_string = add_atomic_props_block(cif_string)

        # Extract species, spacegroup and composition of crystal structure
        composition = Composition(extract_formula_nonreduced(cif_string)).as_dict()
        species = list(set(composition.keys()))
        spacegroup = extract_space_group_symbol(cif_string)

        # Save output to pickle file
        output_dict = {
            'cif_name': cif_name,
            'cif_string': cif_string,
            'strat_key': strat_key,
            'species': species,
            'spacegroup': spacegroup,
            'composition': composition,
        }
        output_path = os.path.join(to_dir, cif_name + '.pkl.gz')
        safe_pkl_gz(output_dict, output_path)

    except Exception as e:
        if debug:
            print(f"Error processing {cif_name}: {e}")

        logger = logging.getLogger()
        logger.exception(f"Exception in worker function pre-processing CIF {cif_name}, with error:\n {e}\n\n")
        
def xrd_worker(args):

    # Extract arguments
    from_path, task_dict, debug, to_dir = args

    # Open pkl and extract
    with gzip.open(from_path, "rb") as f:
        data = pickle.load(f)
    cif_name = data['cif_name']
    cif_string = data['cif_string']

    try:
        # Load structure and parse to ASE
        structure = parser_from_string(cif_string).get_structures()[0]

        # Init calculator object and calculate XRD pattern
        xrd_calc = XRDCalculator(wavelength=task_dict['wavelength'])
        if task_dict['qmax'] >= ((4 * np.pi) / xrd_calc.wavelength) * np.sin(np.radians(180/2)):
            two_theta_range = None
        else:
            tth_min = np.degrees(2 * np.arcsin((task_dict['qmin'] * xrd_calc.wavelength) / (4 * np.pi)))
            tth_max = np.degrees(2 * np.arcsin((task_dict['qmax'] * xrd_calc.wavelength) / (4 * np.pi)))
            two_theta_range = (tth_min, tth_max)
        xrd_pattern = xrd_calc.get_pattern(structure, two_theta_range=two_theta_range)

        # Convert units to Q
        theta = np.radians(xrd_pattern.x / 2)
        q_disc = 4 * np.pi * np.sin(theta) / xrd_calc.wavelength # Q = 4 pi sin theta / lambda
        iq_disc = xrd_pattern.y / (np.max(xrd_pattern.y) + 1e-16)

        output_dict = {
            'cif_name': cif_name,
            'xrd': {
                'q': q_disc,
                'iq': iq_disc,
            },
        }

        output_path = os.path.join(to_dir, cif_name + '.pkl.gz')
        safe_pkl_gz(output_dict, output_path)

    except Exception as e:
        if debug:
            print(f"Error processing {cif_name}: {e}")
        logger = logging.getLogger()
        logger.exception(f"Exception in worker function with disc xrd calculation for CIF with name {cif_name}, with error:\n {e}\n\n")
    
def cif_tokenizer_worker(args):
    
    # Extract arguments
    from_path, _, debug, to_dir = args

    # Open pkl and extract
    with gzip.open(from_path, "rb") as f:
        data = pickle.load(f)
    cif_name = data['cif_name']
    cif_string = data['cif_string']

    try:
        # Remove symmetries and header from cif_string before tokenizing
        cif_string = remove_cif_header(cif_string)
        cif_string_reduced = replace_data_formula_with_nonreduced_formula(cif_string)
        cif_string_nosym = replace_symmetry_loop_with_P1(cif_string_reduced)

        # Initialise Tokenizer
        tokenizer = Tokenizer()
        tokenize = tokenizer.tokenize_cif
        encode = tokenizer.encode

        cif_tokens = encode(tokenize(cif_string_nosym))
    
        # Save output to pickle file
        output_dict = {
            'cif_tokens': cif_tokens,
        }
        output_path = os.path.join(to_dir, cif_name + '.pkl.gz')
        safe_pkl_gz(output_dict, output_path)

    except Exception as e:
        if debug:
            print(f"Error processing {cif_name}: {e}")
        logger = logging.getLogger()
        logger.exception(f"Exception in worker function with tokenization for CIF with name {cif_name}, with error:\n {e}\n\n")

def name_and_strat(path):
    with gzip.open(path, 'rb') as f:
        data = pickle.load(f)
        try:
            cif_name = data['cif_name']
            strat_key = data['strat_key']
            return cif_name, strat_key
        except NameError:
            return None

def load_data_from_data_types_list(path_basename, data_types):
    data_dict = {}
    # Loop through data types
    for dct in data_types:
        file_path = os.path.join(dct['dir'], path_basename + '.pkl.gz')
        with gzip.open(file_path, 'rb') as f:
            data = pickle.load(f)
            for key in dct['keys']:
                data_dict[key] = data[key]
    return data_dict

def save_h5(h5_path, cif_names, data_types):

    with h5py.File(h5_path, 'w') as h5f:
        # Placeholder for datasets
        dsets = {} # Stores datasets for each data type
        current_size = 0

        for idx, name in enumerate(tqdm(cif_names, desc=f'Serializing {h5_path}')):
            # Load data for all data types
            try:
                data_dict = load_data_from_data_types_list(name, data_types)
            except:
                print(f"Error in loading: {name}")
                continue

            # Initialise datasets if processing the first file
            if idx == 0:
                for data_key, data_value in data_dict.items():
                    # Determine the data type and create datasets accordinly
                    if isinstance(data_value, np.ndarray):
                        # For numpy arrays
                        data_shape = data_value.shape
                        data_dtype = data_value.dtype
                        if len(data_shape) == 1:
                            # For one-dimensional arrays
                            max_shape = (None, data_shape[0])
                            initial_shape = (0, data_shape[0])
                        else:
                            # For multi-dimensional arrays
                            max_shape = (None,) + data_shape[1:]
                            initial_shape = (0,) + data_shape[1:]

                        dsets[data_key] = h5f.create_dataset(
                            data_key,
                            shape = initial_shape,
                            maxshape = max_shape,
                            chunks = True,
                            dtype = data_dtype,
                        )
                    elif isinstance(data_value, str):
                        # For strings
                        dt = h5py.string_dtype(encoding='utf-8')
                        dsets[data_key] = h5f.create_dataset(
                            data_key,
                            shape = (0,),
                            maxshape = (None,),
                            dtype = dt,
                        )
                    elif isinstance(data_value, int):
                        # For integers
                        dsets[data_key] = h5f.create_dataset(
                            data_key,
                            shape = (0,),
                            maxshape = (None,),
                            dtype = 'int32',
                        )
                    elif isinstance(data_value, float):
                        # For floats
                        dsets[data_key] = h5f.create_dataset(
                            data_key,
                            shape = (0,),
                            maxshape = (None,),
                            dtype = 'float32',
                        )
                    elif isinstance(data_value, (list, set)):
                        # Determine if the list contains numbers or strings
                        if all(isinstance(item, (int, float, np.number)) for item in data_value):
                            # For lits of numbers
                            if all(isinstance(item, int) for item in data_value):
                                dt = h5py.vlen_dtype(np.dtype('int32'))
                            else:
                                dt = h5py.vlen_dtype(np.dtype('float32'))
                            
                            dsets[data_key] = h5f.create_dataset(
                                data_key,
                                shape = (0,),
                                maxshape = (None,),
                                dtype = dt,
                            )
                        elif all(isinstance(item, str) for item in data_value):
                            # For lists of strings
                            dt = h5py.string_dtype(encoding='utf-8')
                            dsets[data_key] = h5f.create_dataset(
                                data_key,
                                shape = (0,),
                                maxshape = (None,),
                                dtype = dt,
                            )
                        else:
                            raise TypeError(f"Unsupported data type for key '{data_key}': {type(data_value)}")
                    elif isinstance(data_value, dict):
                        # For dicts of arrays
                        # A dataset for each of the arrays in the dict
                        dt = h5py.vlen_dtype(np.dtype('float32'))
                        for key, values in data_value.items():
                            dsets[data_key + '.' + key] = h5f.create_dataset(
                                data_key + '.' + key,
                                shape = (0,),
                                maxshape = (None,),
                                dtype = dt,
                            )
                    else:
                        raise TypeError(f"Unsupported data type for key '{data_key}': {type(data_value)}")

            # Append data to datasets
            for data_key, data_value in data_dict.items():
                if isinstance(data_value, dict):
                    for key, values in data_value.items():
                        dset = dsets[data_key + '.' + key]
                        dset.resize(current_size + 1, axis=0)
                        dset[current_size] = values
                else:
                    dset = dsets[data_key]
                    # Resize dataset to accomodate new data
                    dset.resize(current_size + 1, axis=0)
                    # Assign data based on type
                    if isinstance(data_value, np.ndarray):
                        dset[current_size] = data_value
                    elif isinstance(data_value, (str, int, float)):
                        dset[current_size] = data_value
                    elif isinstance(data_value, (list, set)):
                        if all(isinstance(item, (int, float, np.number)) for item in data_value):
                            # Convert list to numpy array
                            dset[current_size] = np.array(data_value)
                        elif all(isinstance(item, str) for item in data_value):
                            # Serialize the list to a JSON string
                            dset[current_size] = json.dumps(list(data_value))
                        else:
                            raise TypeError(f"Unsupported data type for key '{data_key}': {type(data_value)}")
                    else:
                        raise TypeError(f"Unsupported data type for key '{data_key}': {type(data_value)}")
            current_size += 1

def serialize(root, num_workers, seed, ignore_data_split=False):

    # Locate available data TODO make this automatic based on folder names etc.
    pre_dir = os.path.join(root, "preprocessed")
    pre_paths = glob(os.path.join(pre_dir, "*.pkl.gz"))
    assert len(pre_paths) > 0, f"No preprocessing files found in {pre_dir}"
    dataset_size = len(pre_paths)
    
    # Make output folder
    ser_dir = os.path.join(root, "serialized")
    os.makedirs(ser_dir, exist_ok=True)
    
    # Retrieve all cif names and stratification keys
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(name_and_strat, pre_paths), total=len(pre_paths), desc="Retrieving names and stratification keys", leave=False))

    # Seperate cif neams and stratification keys
    cif_names = [item[0] for item in results if item is not None]
    strat_keys = [item[1] for item in results if item is not None]
    
    # Data types
    data_types = []

    # Preprocessed
    data_types.append({'dir': pre_dir, 'keys': ['cif_name', 'cif_string', 'strat_key', 'species', 'spacegroup']})
    
    xrd_dir = os.path.join(root, "xrd")
    xrd_paths = glob(os.path.join(xrd_dir, "*.pkl.gz"))
    if len(xrd_paths) > 0:
        data_types.append({'dir': xrd_dir, 'keys': ['xrd']})

    cif_token_dir = os.path.join(root, "cif_tokens")
    cif_token_paths = glob(os.path.join(cif_token_dir, "*.pkl.gz"))
    if len(cif_token_paths) > 0:
        data_types.append({'dir': cif_token_dir, 'keys': ['cif_tokens']})
    
    # Create data splits
    if not ignore_data_split:
        train_size = int(0.9 * dataset_size)
        val_size = int(0.075 * dataset_size)
        test_size = dataset_size - train_size - val_size

        print("Train size:", train_size)
        print("Val size:", val_size)
        print("Test size:", test_size)

        cif_names_temp, cif_names_test, strat_keys_temp, _ = train_test_split(
            cif_names, strat_keys, test_size = test_size, stratify = strat_keys, random_state = seed,
        )
        cif_names_train, cif_names_val = train_test_split(
            cif_names_temp, test_size = test_size, stratify = strat_keys_temp, random_state = seed,
        )
    
        for cif_names, split_name in zip([cif_names_train, cif_names_val, cif_names_test], ['train', 'val', 'test']):
            h5_path = os.path.join(ser_dir, f'{split_name}.h5')
            save_h5(h5_path, cif_names, data_types)
    else:
        h5_path = os.path.join(ser_dir, f'test.h5')
        save_h5(h5_path, cif_names, data_types)

def retrieve_worker(args):
    
    # Extract args
    path, key = args

    # Open pkl and extract
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)[key]

    return data

def collect_data(root, get_from, key, num_workers=cpu_count() - 1):

    # Find paths
    paths = glob(os.path.join(root, get_from, "*.pkl.gz"))
    args = [(path, key) for path in paths] 

    # Parallel process retrieving the results
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_workers) as pool:
        key_data = list(tqdm(pool.imap(retrieve_worker, args), total=len(paths), desc=f"Retrieving {key} from {get_from}...", leave=False))
        
    return key_data

if __name__ == "__main__":

    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Prepare custom CIF files and save to a tar.gz file.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, help="Path to the outer data directory", required=True)
    parser.add_argument("--name", "-n", type=str, help="Name of data preparation", required=True)
    parser.add_argument("--strat-group-size", type=int, help="Spacegroup group size for stratification", default=10)
    parser.add_argument("--num-decimal-places", type=int, help="Number of decimal places for floats in CIF files", default=4)
    parser.add_argument("--include-occupancy-structures", help="Include structures with occupancies less than unity", action="store_true")

    parser.add_argument("--preprocess", help="preprocess files", action="store_true")
    parser.add_argument("--xrd", help="calculate XRD patterns", action="store_true")  # Placeholder for future implementation
    parser.add_argument("--tokenize", help="tokenize CIFs", action="store_true")  # Placeholder for future implementation
    parser.add_argument("--serialize", help="serialize data by hdf5 convertion", action="store_true")  # Placeholder for future implementation
    parser.add_argument("--all", help="process, calculate xrd, tokenize and serialize", action="store_true")
    parser.add_argument("--ignore-data-split", help='Ignore data splitting and serialize all data into test.h5', action='store_true')

    parser.add_argument("--debug-max", help="Debug-feature: max number of files to process", type=int, default=0)
    parser.add_argument("--debug", help="Debug-feature: whether to print debug messages", action="store_true")
    parser.add_argument("--num-workers", help="Number of workers for each processing step", type=int, default=0)
    parser.add_argument("--raw-from-gzip", help="Whether raw CIFs come packages in gzip pkl", action="store_true")
    parser.add_argument("--save-species-to-metadata", help="Extraordinary save of species to metadata", action="store_true")

    args = parser.parse_args()

    if args.all:
        args.preprocess = True
        args.xrd = True
        args.tokenize = True
        args.serialize = True

    # Number of multiprocessing workers
    if args.num_workers == 0: # Default
        args.num_workers = cpu_count() - 1
    else:
        args.num_workers = min(cpu_count() - 1, args.num_workers)

    # Make data prep directory and update data_dir
    args.data_dir = os.path.join(args.data_dir, args.name)
    os.makedirs(args.data_dir, exist_ok=True)

    # Adjust debug_max if no limit is specified
    if args.debug_max == 0:
        args.debug_max = None

    if args.preprocess:
        preprocess_dict = {
            'spacegroup_strat_group_size': args.strat_group_size,
            'num_decimal_places': args.num_decimal_places,
            'include_occupancy_structures': args.include_occupancy_structures,
        }
        run_subtasks(
            root = args.data_dir, 
            worker_function = preprocess_worker,
            get_from = "../",
            save_to = "preprocessed",
            task_kwargs_dict = preprocess_dict,
            announcement = "PREPROCESSING",
            debug = True,
            num_workers = args.num_workers,
            from_gzip = args.raw_from_gzip,
            debug_max = args.debug_max,
        )
        
        # Save species to metadata
        species = collect_data(
            root = args.data_dir,
            get_from = "preprocessed",
            key = "species",
        ) # Returns a list of lists
        species = {'species': list(set([item for sublist in species for item in sublist]))}
        save_metadata(species, args.data_dir)

    # If metadata species is not available, extra option for retrieving
    if args.save_species_to_metadata:
        # Save species to metadata
        species = collect_data(
            root = args.data_dir,
            get_from = "preprocessed",
            key = "species",
        ) # Returns a list of lists
        species = {'species': list(set([item for sublist in species for item in sublist]))}
        save_metadata(species, args.data_dir)
    
    if args.xrd:
        xrd_dict = {
            'wavelength': 'CuKa',
            'qmin': 0.0,
            'qmax': 10.0,
        }
        run_subtasks(
            root = args.data_dir, 
            worker_function = xrd_worker,
            get_from = "preprocessed",
            save_to = "xrd",
            task_kwargs_dict = xrd_dict,
            announcement = "CALCULATING XRD",
            debug = args.debug,
            num_workers = args.num_workers,
        )
        save_metadata({'xrd': xrd_dict}, args.data_dir)
    
    if args.tokenize:
        run_subtasks(
            root = args.data_dir, 
            worker_function = cif_tokenizer_worker,
            get_from = "preprocessed",
            save_to = "cif_tokens",
            announcement = "TOKENIZING CIFS",
            debug = args.debug,
            num_workers = args.num_workers,
        )
    
    if args.serialize:
        serialize(args.data_dir, args.num_workers, args.seed, args.ignore_data_split)

    # Store all arguments passed to the main function in centralized metadata
    metadata = {
        "arguments": vars(args)
    }

    # Finalize metadata saving after all processing steps
    save_metadata(metadata, args.data_dir)
