# Minicif SLURM Scripts

Run from the repository root after activating the Python environment.

## 1. Prepare data

From raw `.cif` files:

```bash
sbatch minislurm/prepare_minicif_dataset.sh \
  --raw-dir data/raw_cifs \
  --out-dir data \
  --num-workers 8
```

From a legacy `.pkl.gz` raw bundle:

```bash
sbatch minislurm/prepare_minicif_dataset.sh \
  --raw-dir data/noma \
  --out-dir data \
  --raw-from-gzip \
  --num-workers 8
```

Prepare a deterministic subset from a larger source:

```bash
sbatch minislurm/prepare_minicif_dataset.sh \
  --raw-dir data/noma \
  --out-dir data/minicif_debug \
  --raw-from-gzip \
  --max-samples 10000 \
  --sample-strategy random \
  --num-workers 8
```

Preparation is resumable by default through `OUT_DIR/prep_checkpoint.pkl.gz`.
Use `--no-resume` to ignore an existing checkpoint, or `--checkpoint-path PATH` to store it elsewhere.

The output should contain `data/serialized/train.h5`, `val.h5`, and `test.h5`.

## 2. Train

Default small minicif config:

```bash
sbatch minislurm/train_minicif.sh
```

Custom config:

```bash
sbatch minislurm/train_minicif.sh --config configs/minicif_small_config.yaml
```

## 3. Evaluate and visualize

Default validation/test report:

```bash
sbatch minislurm/evaluate_minicif.sh
```

Custom checkpoint or generation settings:

```bash
sbatch minislurm/evaluate_minicif.sh \
  --checkpoint minicif_model_small/ckpt.pt \
  --dataset-dir data \
  --splits val test \
  --num-reps 8 \
  --generation-batch-size 8
```

The report is written to `CHECKPOINT_DIR/minicif_report` unless `--out-dir` is passed.
