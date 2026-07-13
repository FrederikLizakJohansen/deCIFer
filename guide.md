# Minicif v2 quick guide

Run all commands from the repository root after activating the Python environment.

## 1. Generate the dataset

The default input is the NOMA gzip bundle under `data/noma`. Prepared data is
written to `data/noma_minicif_v2`.

```bash
sbatch minislurm/prepare_minicif_v2_dataset.sh
```

To use different locations:

```bash
RAW_DIR=/path/to/raw/noma \
OUT_DIR=/path/to/noma_minicif_v2 \
sbatch minislurm/prepare_minicif_v2_dataset.sh
```

After preparation, verify the dataset:

```bash
python bin/audit_minicif_v2.py \
  --config configs/minicif_v2_small_config.yaml \
  --max-items 100 \
  --output minicif_v2_preflight.json
```

## 2. Train the model

First run the short GPU integration check:

```bash
sbatch minislurm/train_minicif_v2.sh \
  --config configs/minicif_v2_gpu_smoke_config.yaml
```

Then train the small model:

```bash
sbatch minislurm/train_minicif_v2.sh \
  --config configs/minicif_v2_small_config.yaml
```

For the recommended research baseline, use:

```bash
sbatch minislurm/train_minicif_v2.sh \
  --config configs/minicif_v2_medium_config.yaml
```

Checkpoints are written to the config's `out_dir`, for example
`minicif_v2_model_small/ckpt.pt`.

## 3. Evaluate the model

Evaluate the small model on validation and test splits:

```bash
CHECKPOINT=minicif_v2_model_small/ckpt.pt \
DATASET_DIR=data/noma_minicif_v2 \
OUT_DIR=minicif_v2_model_small/minicif_report \
sbatch minislurm/evaluate_minicif_v2.sh
```

For the medium model:

```bash
sbatch minislurm/evaluate_minicif_v2.sh
```

Evaluation results are written to the selected `OUT_DIR`.
