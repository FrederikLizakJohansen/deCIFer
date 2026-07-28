# Minicif v2 quick guide

Run these commands from the repository root in a Python 3.12 or 3.13
environment.

## 1. Generate and check data

```bash
python -m pip install -e .
sbatch minislurm/prepare_minicif_v2_dataset.sh \
  --xrd-backend braggcalculator
python bin/audit_minicif_v2.py \
  --config configs/minicif_v2_medium_hybrid_artifacts.yaml \
  --max-items 100 \
  --output minicif_v2_preflight.json
```

The preparation job reads `data/noma`, writes `data/noma_minicif_v2`, and
rejects targets longer than 769 tokens before calculating diffraction. It stores
clean sparse Bragg peaks. Full backgrounds, broadening, noise, detector effects,
and other artifacts are generated on the GPU during training, so an existing
clean minicif v2 dataset does not need to be regenerated.

## 2. Train

Run the two-step artifact-path smoke test, then the medium model:

```bash
sbatch minislurm/train_minicif_v2.sh \
  --config configs/minicif_v2_hybrid_artifacts_smoke.yaml
sbatch minislurm/train_minicif_v2.sh \
  --config configs/minicif_v2_medium_hybrid_artifacts.yaml
```

The medium checkpoint is written to
`minicif_v2_model_medium_hybrid_artifacts/ckpt.pt`.

The three PXRD representation variants are:

| Config | PXRD input |
| --- | --- |
| `minicif_v2_medium_config.yaml` | sparse q positions and peak intensities |
| `minicif_v2_medium_dense_artifacts.yaml` | continuous PXRD intensity trace |
| `minicif_v2_medium_hybrid_artifacts.yaml` | continuous trace plus sparse peaks |

Train the dense-only member with:

```bash
sbatch minislurm/train_minicif_v2.sh \
  --config configs/minicif_v2_medium_dense_artifacts.yaml
```

An additional encoder-allocation experiment assigns about 4.1M of its 7.7M
parameters to a hierarchical dense-PXRD encoder and uses a compact three-layer
minicif decoder:

```bash
sbatch minislurm/train_minicif_v2.sh \
  --config configs/minicif_v2_dense_encoder_heavy_artifacts.yaml
```

Compare the dense-only and encoder-heavy configs to isolate parameter allocation.
Compare the sparse, dense-only, and hybrid configs to study PXRD representation.

Optional standalone PXRD encoder pretraining:

```bash
sbatch minislurm/pretrain_pxrd_encoder.sh \
  --config configs/minicif_pxrd_encoder_pretrain_bragg_artifacts.yaml
```

## 3. Evaluate

Use the fixed-seed artifact profile for repeatable artifact evaluation:

```bash
CHECKPOINT=minicif_v2_model_medium_hybrid_artifacts/ckpt.pt \
DATASET_DIR=data/noma_minicif_v2 \
OUT_DIR=minicif_v2_model_medium_hybrid_artifacts/minicif_report \
sbatch minislurm/evaluate_minicif_v2.sh \
  --artifact-config configs/xrd_artifacts/full_evaluation.yaml
```

Remove `--artifact-config` to evaluate with clean simulated conditions.

To export combined reference/generated PXRD and structure examples:

```bash
python bin/visualize_minicif.py \
  --checkpoint minicif_v2_model_medium/ckpt.pt \
  --dataset-dir data/noma_minicif_v2 \
  --out-dir minicif_v2_model_medium/minicif_report \
  --splits test \
  --prompt-modes pxrd-elements \
  --num-reps 8 \
  --plot-examples 6
```

Figures are written to `OUT_DIR/examples/test/`. Each figure uses the valid
candidate with the lowest Rwp for that sample and prompt mode.

To plot examples from an evaluation that has already finished:

```bash
python bin/plot_minicif_examples.py \
  --report-dir minicif_v2_model_medium/minicif_report \
  --num-examples 8 \
  --selection random \
  --splits test \
  --prompt-modes pxrd-elements
```

This reads the existing evaluation CSV and summary JSON. It does not load the
model or generate new structures. Figures and an `examples.csv` manifest are
written under `REPORT_DIR/evaluation_examples/`.
