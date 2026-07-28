# Minicif v2 quick guide

Run these commands from the repository root in a Python 3.12 or 3.13
environment.

## 1. Generate and check data

```bash
python -m pip install -e .
sbatch minislurm/prepare_minicif_v2_dataset.sh \
  --xrd-backend braggcalculator
python bin/audit_minicif_v2.py \
  --config configs/config_v2/hybrid/standard/medium.yaml \
  --max-items 100 \
  --output minicif_v2_preflight.json
```

Preparation reads `data/noma`, writes `data/noma_minicif_v2`, and rejects
targets longer than 769 tokens before diffraction calculation. The dataset
stores clean sparse Bragg peaks. Dense-profile artifacts are generated during
training.

## 2. Train

Run the two-step artifact smoke test:

```bash
sbatch minislurm/train_minicif_v2.sh \
  --config configs/config_v2/smoke/hybrid.yaml
```

Production configs follow:

```text
configs/config_v2/{peak,dense,hybrid}/{standard,heavy}/{small,medium,large}.yaml
```

For example:

```bash
# Sparse Fourier peaks, standard medium decoder
sbatch minislurm/train_minicif_v2.sh \
  --config configs/config_v2/peak/standard/medium.yaml

# Continuous PXRD, encoder-heavy medium model
sbatch minislurm/train_minicif_v2.sh \
  --config configs/config_v2/dense/heavy/medium.yaml

# Continuous PXRD plus sparse peaks, encoder-heavy large model
sbatch minislurm/train_minicif_v2.sh \
  --config configs/config_v2/hybrid/heavy/large.yaml
```

Checkpoints are organized under:

```text
models/minicif_v2/{representation}/{allocation}/{size}/ckpt.pt
```

Dense and hybrid configs sample the full BraggCalculator artifact profile from
`configs/xrd_artifacts/full_training.yaml`. Peak configs apply q-position and
intensity perturbations to sparse peaks.

## 3. Evaluate

Evaluate the default peak-standard-medium model:

```bash
sbatch minislurm/evaluate_minicif_v2.sh
```

Evaluate another model with deterministic artifacts:

```bash
CHECKPOINT=models/minicif_v2/hybrid/standard/medium/ckpt.pt \
DATASET_DIR=data/noma_minicif_v2 \
OUT_DIR=models/minicif_v2/hybrid/standard/medium/minicif_report \
sbatch minislurm/evaluate_minicif_v2.sh \
  --artifact-config configs/xrd_artifacts/full_evaluation.yaml
```

Omit `--artifact-config` for clean simulated conditions.

Evaluation writes overall metrics, per-crystal-system metrics, Rwp distributions
and CDFs, an Rwp-versus-RMSD figure, and a linear-scale learning curve. It also
exports one reference/generated PXRD-plus-structure example for every available
crystal system by default. Pass `--plot-examples 0` to disable example figures.

Run evaluation directly with a specific prompt:

```bash
python bin/visualize_minicif.py \
  --checkpoint models/minicif_v2/peak/standard/medium/ckpt.pt \
  --dataset-dir data/noma_minicif_v2 \
  --out-dir models/minicif_v2/peak/standard/medium/minicif_report \
  --splits test \
  --prompt-modes pxrd-elements \
  --num-reps 8
```

To plot examples from an existing evaluation:

```bash
python bin/plot_minicif_examples.py \
  --report-dir models/minicif_v2/peak/standard/medium/minicif_report \
  --splits test \
  --prompt-modes pxrd-elements
```
