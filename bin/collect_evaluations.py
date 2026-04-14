#!/usr/bin/env python3

import os
import gzip
import pickle
import argparse
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count

from decifer.evaluation import summarize_successful_evaluation_row

def process_file(file_path):
    """Processes a single .pkl.gz file."""
    try:
        with gzip.open(file_path, 'rb') as f:
            row = pickle.load(f)
        return summarize_successful_evaluation_row(row)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def process(folder, debug_max=None) -> pd.DataFrame:
    """Processes all files in the given folder using multiprocessing."""
    # Get list of files
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pkl.gz')]
    if debug_max is not None:
        files = files[:debug_max]

    # Use multiprocessing Pool to process files in parallel
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, files), total=len(files), desc="Processing files..."))

    # Filter out None results and convert to DataFrame
    data_list = [res for res in results if res is not None]
    return pd.DataFrame(data_list)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-folder-paths", nargs='+', required=True, help="Provide a list of folder paths")
    parser.add_argument("--output-folder", type=str, default='.')
    parser.add_argument("--debug_max", type=int, default=0)
    args = parser.parse_args()
    if args.debug_max == 0:
        args.debug_max = None
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # Loop over folders
    folder_names = [path.split("/")[-1] for path in args.eval_folder_paths]
    for label, path in zip(folder_names, args.eval_folder_paths):
        df = process(path, args.debug_max)
        pickle_path = os.path.join(args.output_folder, label + '.pkl.gz')
        df.to_pickle(pickle_path)
