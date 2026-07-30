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

To refine every generated candidate with BraggCalculator 0.4.1:

```bash
python bin/visualize_minicif.py \
  --checkpoint models/minicif_v2/peak/standard/medium/ckpt.pt \
  --dataset-dir data/noma_minicif_v2 \
  --out-dir models/minicif_v2/peak/standard/medium/minicif_report \
  --splits test \
  --prompt-modes pxrd-elements \
  --num-reps 8 \
  --refinement-config configs/refinement/quick.yaml
```

The report compares initial and refined metrics and Rwp distributions globally
and by crystal system. Full per-candidate results are stored in
`minicif_refinement_results.jsonl.gz`. See
`configs/refinement/README.md` for two-theta/Q, X-ray/neutron, policy,
parameter-group, device, wavelength, and species-assignment settings.

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

Evaluation progress is committed after each dataset sample and prompt mode under
`OUT_DIR/evaluation_checkpoints/{split}.sqlite3`. Rerun the same command with
the same output directory to resume. Configuration changes that affect results
must use a new output directory or `--restart-splits`.

Train, validation, and test can be evaluated as separate jobs:

```bash
python bin/visualize_minicif.py \
  --checkpoint models/minicif_v2/peak/standard/medium/ckpt.pt \
  --dataset-dir data/noma_minicif_v2 \
  --out-dir models/minicif_v2/peak/standard/medium/minicif_report \
  --splits train \
  --prompt-modes pxrd-elements \
  --num-reps 8

# Run the same command later with --splits val, then --splits test.
```

Each run rebuilds the combined report from every split checkpoint in the output
directory. Per-split tables are written to `split_metrics/`. Rebuild the
combined CSV, JSON, refinement export, and figures without loading the model:

```bash
python bin/visualize_minicif.py \
  --combine-only \
  --out-dir models/minicif_v2/peak/standard/medium/minicif_report
```

To discard and recompute one split, run its evaluation command with
`--restart-splits --splits SPLIT`.

To plot examples from an existing evaluation:

```bash
python bin/plot_minicif_examples.py \
  --report-dir models/minicif_v2/peak/standard/medium/minicif_report \
  --splits test \
  --prompt-modes pxrd-elements \
  --show-refined
```

`--show-refined` requires an evaluation run with refinement enabled. Each figure
then compares reference, generated, and refined PXRD profiles and structures.
Available evaluation presets are `quick.yaml`, `cautious.yaml`, `robust.yaml`,
`cautious_coordinates.yaml`, and `cautious_species_assignment.yaml` under
`configs/refinement/`. Use a separate report directory for each preset.
