# Minicif SLURM Scripts

Run from the repository root after activating the Python environment.

## 1. Prepare data

From raw `.cif` files:

```bash
sbatch minislurm/prepare_minicif_dataset.sh \
  --raw-dir data/raw_cifs \
  --out-dir data/minicif \
  --num-workers 8
```

From a legacy `.pkl.gz` raw bundle:

```bash
sbatch minislurm/prepare_minicif_dataset.sh \
  --raw-dir data/noma \
  --out-dir data/minicif \
  --raw-from-gzip \
  --num-workers 8
```

The output should contain `data/minicif/serialized/train.h5`, `val.h5`, and `test.h5`.

## 2. Train

Default small minicif config:

```bash
sbatch minislurm/train_minicif.sh
```

Custom config:

```bash
sbatch minislurm/train_minicif.sh --config configs/minicif_small_config.yaml
```

Evaluation scripts are intentionally not included yet; minicif generation still needs a minicif-to-CIF rendering path before the existing CIF validity/evaluation workflow applies.
