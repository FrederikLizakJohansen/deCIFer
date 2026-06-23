# Tackling Real-World Crystal Structure Prediction from Powder X-ray Diffraction Data

**Authors:** Frederik Lizak Johansen, Adam F. Sapnik, Erik Bjørnager Dam, Raghavendra Selvan, Kirsten M. Ø. Jensen

This folder contains the scripts, configs, and notebooks for the follow-up paper that evaluates deCIFer on real-world synchrotron PXRD data and tests its robustness to common measurement artefacts.

All scripts must be run from the **repo root**, not from inside this folder. See the main [README](../README.md) for setup and installation.

---

## Contents

```
follow-up-paper-Tackling-Real-World-CSP/
├── figure-scripts/          # Scripts that generate all paper figures
├── configs-TRW-CSP-PXRD/   # YAML configs for ablation sweeps and training
└── gen-figures-TRW-CSP-PXRD.ipynb  # Interactive notebook (same figures as scripts)
```

### Downloads

All files are available from the frozen archive:

➡️ **[deCIFer Data Archive](https://www.erda.dk/archives/b7342461e7c932bd99e8273c6a49e97b/published-archive.html)**

| File | Contents |
|---|---|
| `TRW-CSP-PXRD-data.zip` | Experimental PXRD scans and precomputed result pickles |
| `decifer_v1_ckpt.pt` | deCIFer model checkpoint |
| `u-decifer_v1_ckpt.pt` | U-deCIFer model checkpoint |
| `noma.zip` | NOMA dataset, pre-serialized HDF5 splits |
| `noma_cifs_raw.pkl.gz` | NOMA dataset, raw CIFs (only needed to re-run preprocessing) |

**Extracting `TRW-CSP-PXRD-data.zip`:** place it at the repo root and extract so that the layout looks like:

```
TRW-CSP-PXRD-data/
├── exp-data/                # Raw experimental PXRD scans (.xy / .xye)
│   ├── Si.xy
│   ├── crystalline_CeO2.xye
│   ├── nanoparticle_CeO2.xy
│   └── Fe2O3.xy
└── pickles/                 # Precomputed result pickles
    ├── crystalline_CeO2.pkl     # experimental protocol results, CeO2
    ├── crystalline_Si.pkl       # experimental protocol results, Si
    ├── crystalline_Fe2O3.pkl    # experimental protocol results, Fe2O3
    ├── particles_CeO2.pkl       # experimental protocol results, nanoparticle CeO2
    ├── particles_CeO2_ID5.pkl   # alternative nanoparticle CeO2 run
    ├── ablation_mainfig_cubic_large.pkl          # deCIFer cubic ablation (main text)
    ├── ablation_mainfig_cubic_large_nocond.pkl   # U-deCIFer cubic ablation (main text)
    ├── ablation_mainfig_hexagonal_large_FeO.pkl
    ├── ablation_mainfig_hexagonal_large_nocond_FeO.pkl
    ├── ablation_mainfig_trigonal_large_FeO.pkl
    ├── ablation_mainfig_trigonal_large_nocond_FeO.pkl
    ├── ablation_appendix_*.pkl  # extended appendix ablation runs
    └── ...
```

`pickles/` contains the final results from all experiments reported in the paper. It does not contain the NOMA dataset. Reproducing figures only requires these pickles. Running new ablations requires the NOMA dataset (see below).

---

## Reproducing the Paper Figures

Once the data download is extracted to `TRW-CSP-PXRD-data/` (see above), the figure scripts read directly from `TRW-CSP-PXRD-data/pickles/`. No model or NOMA dataset is needed to regenerate the figures.

### Running the figure scripts

Run all figures at once:

```bash
python follow-up-paper-Tackling-Real-World-CSP/figure-scripts/generate_all_updated_figures.py
```

Output goes to `revision-final-figures/generated/` at the repo root:
- `generated/experimental/` -- experimental-results figures
- `generated/analysis/` -- ablation and robustness figures
- `generated/ranking-sensitivity/` -- R_wp ranking-sensitivity figures

Individual scripts can also be run directly and each accepts `--help`:

| Script | Output |
|---|---|
| `experimental_appendix_figures.py` | Per-material appendix figures |
| `selected_conditions_figure.py` | Main experimental figure |
| `selected_conditions_v2.py` | Main experimental figure (v2) |
| `experimental_results_figure.py` | Combined experimental figure |
| `ablation_figure_exports.py` | Ablation figures (main + appendix) |
| `robustness_figure_exports.py` | DeltaMR and DeltaRD robustness summaries |
| `polymorph_figure_exports.py` | FeO2 polymorph figure |
| `rwp_ranking_sensitivity.py` | R_wp ranking-sensitivity heatmaps |

The interactive notebook `gen-figures-TRW-CSP-PXRD.ipynb` covers the same figures.

---

## Running Experiments from Scratch

### Model checkpoints

Running `bin/ablation.py` or `bin/run_protocol.py` requires the model checkpoints. The ablations compare deCIFer against U-deCIFer, so both are needed for the paired DeltaMR / DeltaRD metrics. Download from the [archive](https://www.erda.dk/archives/b7342461e7c932bd99e8273c6a49e97b/published-archive.html) and place the checkpoints as follows:

```bash
mkdir -p deCIFer_model U-deCIFer_model
mv decifer_v1_ckpt.pt deCIFer_model/ckpt.pt
mv u-decifer_v1_ckpt.pt U-deCIFer_model/ckpt.pt
```

### Preparing the NOMA dataset for ablations

`bin/ablation.py` requires the NOMA test split at `data/noma/serialized/test.h5`. Two download options are available:

**Option A -- pre-serialized (recommended):**

Download `noma.zip` from the [archive](https://www.erda.dk/archives/b7342461e7c932bd99e8273c6a49e97b/published-archive.html) and extract it at the repo root. It should produce:

```
data/noma/
└── serialized/
    ├── train.h5
    ├── val.h5
    └── test.h5
```

No further preparation is needed. The ablation configs already point to `data/noma/serialized/test.h5`.

**Option B -- raw CIFs:**

Download `noma_cifs_raw.pkl.gz` from the [archive](https://www.erda.dk/archives/b7342461e7c932bd99e8273c6a49e97b/published-archive.html), place it at `data/noma/noma_cifs_raw.pkl.gz`, then run:

```bash
python bin/prepare_dataset.py \
  --data-dir data/noma/ \
  --name noma \
  --all \
  --raw-from-gzip
```

This writes the HDF5 files to `data/noma/serialized/`. The NOMA dataset is assembled from [Materials Project](https://materialsproject.org/), [OQMD](https://oqmd.org/), and [NOMAD](https://nomad-lab.eu/). See the main README [Data Preparation](../README.md#data-preparation) section for the full argument reference.

### Running the ablation

The ablation runs one model at a time. To reproduce the paired comparisons from the paper, run it once for deCIFer and once for U-deCIFer:

```bash
# deCIFer -- cubic polymorph (main text)
python bin/ablation.py --config follow-up-paper-Tackling-Real-World-CSP/configs-TRW-CSP-PXRD/ablation_config_cluster.yaml

# U-deCIFer -- cubic polymorph (main text)
python bin/ablation.py --config follow-up-paper-Tackling-Real-World-CSP/configs-TRW-CSP-PXRD/ablation_config_cluster_udecifer.yaml
```

The `_cluster` configs use `batch_size: 10` for multi-GPU runs. The `_local` configs use `batch_size: 1` for single-machine use. Results are saved as pickle files; the figure scripts load and pair the conditioned / unconditioned pickles to compute DeltaMR and DeltaRD.

### Running the experimental PXRD protocols

`bin/run_protocol.py` runs deCIFer on the CeO2 nanoparticle scans from the paper under several conditioning protocols. The raw scans are in `TRW-CSP-PXRD-data/exp-data/`.

```bash
python bin/run_protocol.py \
  --model-path deCIFer_model/ckpt.pt \
  --zip-path TRW-CSP-PXRD-data/exp-data/ \
  --n-trials 25 \
  --suffix run1
```

For custom PXRD data, use `DeciferPipeline` from `bin/experimental_pipeline.py` directly:

```python
from bin.experimental_pipeline import DeciferPipeline

pipeline = DeciferPipeline(
    model_path="deCIFer_model/ckpt.pt",
    zip_path="TRW-CSP-PXRD-data/exp-data/",
    temperature=1.0,
    max_new_tokens=3000,
    results_output_folder="results/",
)

pipeline.prepare_target_data(
    target_file="crystalline_CeO2.xye",
    wavelength=None,   # None if data is already in Q; provide wavelength in Angstrom if in 2theta
    q_min_crop=1.5,
    q_max_crop=8.0,
)

pipeline.run_experiment_protocol(
    n_trials=25,
    composition="Ce4O8",
    spacegroup="Fm-3m_sg",
    crystal_systems=[7],
    save_to="ceo2_Fm-3m.pkl",
    protocol_name="Ce4O8_Fm-3m",
)
```

Preprocessing applied to all experimental patterns: conversion from 2theta to Q if wavelength is given, interpolation onto Q = 0--10 inverse Angstrom at 1000 points, normalization to unit maximum. No denoising, smoothing, or background subtraction is applied.

---

## Training from Scratch

Training is only needed to reproduce the model weights. To reproduce figures or run protocols, use the pre-trained checkpoint.

**deCIFer** (conditioned, `condition: True`):

```bash
python bin/train.py --config follow-up-paper-Tackling-Real-World-CSP/configs-TRW-CSP-PXRD/deCIFer_NOMA_training_config.yaml
```

**U-deCIFer** (unconditioned baseline, `condition: False`):

```bash
python bin/train.py --config follow-up-paper-Tackling-Real-World-CSP/configs-TRW-CSP-PXRD/U-deCIFer_NOMA_training_config.yaml
```

Both configs use `dataset: 'data/noma/full'` and save checkpoints to `deCIFer_model/` and `U-deCIFer_model/` respectively. A GPU is required. See the main README [Training](../README.md#training) section for full details.
