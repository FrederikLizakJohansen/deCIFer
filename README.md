[![Static Badge](https://img.shields.io/badge/arXiv-2502.02189-brightgreen?style=flat)](https://arxiv.org/abs/2502.02189)

# deCIFer: Crystal Structure Prediction from Powder Diffraction Data

## Table of Contents
1. [Setup](#setup)
2. [Data Preparation](#data-preparation)
3. [Training](#training)
4. [Evaluation Pipeline](#evaluation)
5. [CIF Generation Consistency Experiment](#cif-generation-consistency-experiment)
6. [License](#license)

## Setup
We recommend using **Python 3.9**, as deCIFer was developed and tested with this version. Other versions may work but have not been verified.

1. **Clone the repository**:
```bash
git clone https://github.com/XXXX/deCIFer.git
cd deCIFer
```

2. **Install deCIFer using setup.py.** We recommend an isolated environment (e.g., with Conda or venv):
```bash
pip install -e .
```

3. **Ensure that you have PyTorch installed:**
Follow the instructions on the official PyTorch website to install the appropriate version for your system: PyTorch Installation Guide. (https://pytorch.org/get-started/locally/)

4. **Install other dependencies**:
```bash
pip install numpy pandas matplotlib seaborn pyYAML tqdm omegaconf h5py pymatgen periodictable scikit-learn notebook
```

## Data Preparation
Before training or evaluation, the dataset must be preprocessed into a structured format that deCIFer can use. This includes **parsing CIF files**, **computing XRD patterns**, **tokenizing CIFs**, and **serializing the processed data into HDF5 format**.
To prepare a dataset, use the `prepare_dataset.py` script with the desired options. Here is an example:
```bash
python bin/prepare_dataset.py --data-dir data/noma/ --name noma-1k --debug-max 1000 --all --raw-from-gzip
```
### Arguments:

- **General options**:
  - `--data-dir <path>`: Path to the directory containing raw CIF and PXRD data.
  - `--name <str>`: Identifier for the dataset (used to create an organized structure).
  - `--debug-max <int>`: Limits processing to the first `N` samples (useful for debugging).
  - `--raw-from-gzip`: If raw CIFs are stored in a `.gz` archive, extract them before processing. (This is the case for the NOMA dataset, depeding on how the data is downloaded)

- **Processing steps**:
  - `--preprocess`: Parses and cleans CIF files.
  - `--xrd`: Computes diffraction patterns.
  - `--tokenize`: Tokenizes CIF files for transformer-based models.
  - `--serialize`: Serializes the processed dataset into HDF5 format.
  - `--all`: Runs **all** preprocessing steps in sequence.

- **Processing options**:
  - `--num-workers <int>`: Number of parallel processes to use (default: all available CPUs - 1).
  - `--include-occupancy-structures`: Include structures with atomic site occupancies below 1. (False for deCIFer and U-deCIFer)
  - `--ignore-data-split`: Disable automatic train/val/test splitting and serialize all data into `test.h5`.

### Output Directory Structure

After running `prepare_dataset.py`, the processed data will be stored in the following structure:
```bash
data/noma-1k/
├── preprocessed/ – Parsed CIF files (cleaned, formatted, tokenized)
├── xrd/ – Computed XRD patterns
├── cif_tokens/ – Tokenized CIF representations
├── serialized/ – Final dataset (train/val/test) stored as HDF5 files
├── metadata.json – Stores some metadata
```

## Training
### Training From Scratch

To train deCIFer from scratch, you need to specify the training configuration and dataset. The model will initialize randomly and train from the beginning.

#### Required Arguments:
- `--config <path>`: Path to a YAML configuration file specifying training parameters.

#### Example Usage:
```bash
python bin/train.py --config config/train.yaml
```
This initializes a new model and starts training using the settings specified in `train.yaml`.

### Resuming Training

If training was previously interrupted or stopped, you can resume from a saved checkpoint. This allows training to continue from where it left off.
#### Required Arguments:
- `--config <path>`: Path to the same YAML configuration file used during training.
- Ensure that the `init_from` option in the YAML file is set to `"resume"`.

### Configuration File (`train.yaml`)

The training parameters are stored in a YAML file. Below is an example configuration:

```yaml
out_dir: "models/deCIFer"  # Directory where checkpoints and logs will be saved
dataset: "data/noma/1k/serialized/"  # Path to the dataset
init_from: "scratch"  # Options: 'scratch' (new training), 'resume' (continue training)

# Model parameters
n_layer: 8
n_head: 8
n_embd: 512
dropout: 0.0
boundary_masking: True
condition: True

# Training parameters
batch_size: 64
gradient_accumulation_steps: 40
max_iters: 50000
learning_rate: 6e-4
weight_decay: 0.1
warmup_iters: 2000
early_stopping_patience: 50

# Evaluation settings
eval_interval: 250
eval_iters_train: 200
eval_iters_val: 200
validate: True
```
Modify this file as needed to adjust the training settings.

### Output Structure
After training starts, the model and logs will be saved in the output directory:
```bash
models/deCIFer/
├── ckpt.pt – Latest model checkpoint including model dictionary, loss metrics, etc.
```
### Monitoring Training
During training, logs will be printed to the console, showing:
- Loss values
- Evaluation results on the validation set
- Learning rate adjustments
- Checkpoint saving status
  
Example output:
```bash
iter 1000: loss 0.4123, time 58.2ms
step 1000: train loss 0.4123, val loss 0.4289
saving checkpoint to models/deCIFer/...
```

### Early Stopping
If validation loss does not improve for a set number of evaluations (early_stopping_patience), training will stop automatically.

#### Example Usage:
```bash
python bin/train.py --config config/train.yaml
```
The script will automatically load the latest checkpoint from the output directory specified in the configuration file and continue training.

## Evaluation Pipeline
The evaluation process consists of two main steps: **generating evaluations** for model predictions and **collecting** them into a single file for visualization and analysis.

### Step 1: Generate Evaluations

This step evaluates the model on a test dataset and saves individual `.pkl.gz` evaluation files. These separate files allow evaluations to be merged across different datasets and enable checkpointing (resuming evaluation later if needed).

#### Required Arguments:
- `--model-ckpt <path>`: Path to the trained model checkpoint.
- `--dataset-path <path>`: Path to the test dataset in HDF5 format.
- `--dataset-name <str>`: Identifier for the dataset.
- `--out-folder <path>`: Folder where evaluation files will be stored.

#### Optional Arguments:
- `--num-workers <int>`: Number of worker processes for parallel evaluation.
- `--debug-max <int>`: Maximum number of samples to process (for debugging).
- `--add-composition`: Include atomic composition information in the evaluation.
- `--add-spacegroup`: Include space group information in the evaluation.
- `--max-new-tokens <int>`: Maximum number of tokens to generate for CIF structures.
- `--num-reps <int>`: Number of times to generate a CIF for each test sample.
- `--override`: Force regeneration of evaluations even if files already exist.
- `--temperature <float>`: Sampling temperature for CIF generation.
- `--top-k <int>`: Top-k filtering during sampling.
- `--condition`: Use XRD conditioning for generation.

#### Example Usage:

Run evaluation on a test dataset and save results:

```bash
python bin/evaluate.py \
  --model-ckpt deCIFer_model/ckpt.pt \
  --dataset-path data/noma/1k/serialized/test.h5 \
  --dataset-name noma-1k \
  --out-folder eval_files \
  --add-composition \
  --add-spacegroup
```
Each evaluated structure will be saved as an individual .pkl.gz file in eval_files/eval_files/noma-1k/.

### Step 2: Collect Evaluations

Once all evaluations are generated, they need to be aggregated into a single `.pkl.gz` file. This file is used for computing metrics and visualizing results.

#### Required Arguments:
- `--eval-folder-path <path>`: Path to the directory containing the individual evaluation files.
- `--output-file <path>`: Name of the final collected evaluation file.

#### Example Usage:
```bash
python bin/collect_evaluations.py --eval-folder-path eval_files/eval_files/noma-1k --output-file eval_files/noma-1k_collected.pkl.gz
```
This collects all individual `.pkl.gz` evaluation files and stores the merged results in `noma-1k_collected.pkl.gz`.

### Output Structure

After running the evaluation pipeline, the output structure will be:
```bash
eval_files/  
├── eval_files/noma-1k/ – Individual evaluation files (`.pkl.gz` per sample)  
├── noma-1k_collected.pkl.gz – Merged evaluation results
```

#### Evaluation Details

Each evaluation file contains:
- **Generated CIF file**
- **PXRD-based metrics**
- **Validity checks**:
  - Formula consistency
  - Bond length reasonableness
  - Space group consistency
  - Unit cell parameters
- **Root Mean Square Deviation (RMSD)** from reference structure

## CIF Generation Consistency Experiment

The **CIF Generation Consistency Experiment** evaluates the reproducibility of generated crystal structures across multiple repetitions. The script takes a dataset of CIFs, generates multiple versions of each, and compares their structural and diffraction consistency.

### Running the Consistency Experiment

To perform the experiment, use the following command:
```bash
python bin/generation_consistency.py --num_cifs 100 --num_reps 5 --batch_size 16 --qmin 0.0 --qmax 10.0 --qstep 0.01 --fwhm 0.05 --output_folder results/consistency_experiment --model_path deCIFer_model/ckpt.pt --dataset_path data/noma/1k/serialized/test.h5 --add_comp --add_spg
```

#### Required Arguments:
- `--num_cifs <int>`: Number of CIFs to process.
- `--num_reps <int>`: Number of times to generate each CIF.
- `--output_folder <path>`: Directory where results will be saved.
- `--model_path <path>`: Path to the pretrained model checkpoint.
- `--dataset_path <path>`: Path to the dataset in HDF5 format.

#### Optional Arguments:
- `--batch_size <int>`: Number of CIFs to generate in parallel (default: 16).
- `--qmin <float>`: Minimum Q value for XRD computation (default: 0.0).
- `--qmax <float>`: Maximum Q value for XRD computation (default: 10.0).
- `--qstep <float>`: Step size for Q values (default: 0.01).
- `--fwhm <float>`: Full-width at half-maximum for XRD broadening (default: 0.05).
- `--noise <float>`: Optional noise level in XRD computation.
- `--add_comp`: Include atomic composition in the conditioning prompt.
- `--add_spg`: Include space group information in the conditioning prompt.

### Output Structure

After running the experiment, the results will be stored in the following structure:
```bash
results/consistency_experiment/
├── CIF_NAME_1/ – Folder containing generated CIFs for the first structure  
│   ├── CIF_NAME_1_0.cif – First repetition  
│   ├── CIF_NAME_1_1.cif – Second repetition  
│   ├── ...  
├── CIF_NAME_2/ – Folder containing generated CIFs for the second structure  
│   ├── CIF_NAME_2_0.cif  
│   ├── CIF_NAME_2_1.cif 
│   ├── ...  
├── results.pkl – A pickled summary of all results
```

### Evaluation Metrics

Each generated CIF is analyzed for structural and diffraction consistency. The following metrics are computed:

- **Residual Weighted Profile (Rwp)**: Measures the difference between the experimental and generated XRD patterns.
- **Root Mean Square Deviation (RMSD)**: Measures the structural deviation between generated and reference CIFs.
- **Crystal System Consistency**: Checks if the generated structure belongs to the same crystal system as the reference.
- **Space Group Consistency**: Compares the space group number of the generated structure with the original.
- **Lattice Parameter Deviations**:
  - `a`, `b`, `c`: Unit cell lengths.
  - `α`, `β`, `γ`: Unit cell angles.

## License
deCIFer is released under the **MIT License**.
