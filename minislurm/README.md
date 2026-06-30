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
  --raw-from-gzip
```

Parallel shard preparation with a SLURM array:

```bash
sbatch --array=0-31 minislurm/prepare_minicif_dataset.sh \
  --raw-dir data/noma \
  --out-dir data \
  --raw-from-gzip
```

Each array task writes its own checkpoint under `OUT_DIR`. After all array tasks finish, merge the shard checkpoints and write final `train.h5`, `val.h5`, and `test.h5`:

```bash
sbatch minislurm/prepare_minicif_dataset.sh \
  --raw-dir data/noma \
  --out-dir data \
  --raw-from-gzip \
  --num-shards 32 \
  --merge-shards
```

The wrapper uses `$SLURM_CPUS_PER_TASK` as `--num-workers` unless you pass `--num-workers` explicitly. It also defaults to `--chunksize 8`.

Prepare a deterministic subset from a larger source:

```bash
sbatch minislurm/prepare_minicif_dataset.sh \
  --raw-dir data/noma \
  --out-dir data/minicif_debug \
  --raw-from-gzip \
  --max-samples 10000 \
  --sample-strategy random
```

Preparation is resumable by default through `OUT_DIR/prep_checkpoint.pkl.gz`.
Use `--no-resume` to ignore an existing checkpoint, or `--checkpoint-path PATH` to store it elsewhere.
The SLURM script requests a `SIGTERM` five minutes before walltime; the prep script catches it, writes the checkpoint, and exits without rewriting partial HDF5 splits.
You can also stop before walltime explicitly:

```bash
sbatch minislurm/prepare_minicif_dataset.sh \
  --raw-dir data/noma \
  --out-dir data \
  --raw-from-gzip \
  --num-workers 8 \
  --max-runtime-seconds 82800
```

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

Run the small conditioning ablation configs sequentially on one GPU:

```bash
sbatch minislurm/run_minicif_condition_ablation.sh
```

Run only selected ablation configs:

```bash
sbatch minislurm/run_minicif_condition_ablation.sh \
  --configs configs/minicif_condition_ablation/small_mlp_insert.yaml \
            configs/minicif_condition_ablation/small_hybrid_cross.yaml
```

Pretrain the PXRD condition encoder contrastively:

```bash
sbatch minislurm/pretrain_pxrd_encoder.sh
```

Use a custom pretraining config:

```bash
sbatch minislurm/pretrain_pxrd_encoder.sh \
  --config configs/minicif_pxrd_encoder_pretrain_hybrid.yaml
```

The checkpoint is written to:

```text
minicif_pxrd_encoder_pretrain_hybrid/pxrd_encoder_pretrain.pt
```

Live diagnostics are updated during training:

```text
minicif_pxrd_encoder_pretrain_hybrid/contrastive_live.png
minicif_pxrd_encoder_pretrain_hybrid/latest_metrics.json
minicif_pxrd_encoder_pretrain_hybrid/contrastive_metrics.csv
```

For local CUDA laptop debugging, start with the synthetic probe:

```bash
PYTHONPATH=. python bin/pretrain_pxrd_encoder.py \
  --config configs/minicif_pxrd_encoder_pretrain_synthetic_debug.yaml
```

This bypasses HDF5 completely. If it works, test real serialized data with the conservative smoke config:

```bash
PYTHONPATH=. python bin/pretrain_pxrd_encoder.py \
  --config configs/minicif_pxrd_encoder_pretrain_hybrid_laptop_smoke.yaml
```

If the default pretraining config segfaults locally, set `num_workers_dataloader: 0` first and keep `pin_memory: False`. As an opt-in diagnostic fallback, use `preload_dataset_to_memory: True`. HDF5-backed datasets can crash with PyTorch worker multiprocessing on some systems.

Single-node multi-GPU training uses PyTorch DDP automatically when Slurm exposes more than one GPU:

```bash
sbatch --gres=gpu:a100:4 minislurm/train_minicif.sh \
  --config configs/minicif_large_config.yaml
```

To keep the same effective batch size, divide `gradient_accumulation_steps` by the number of GPUs. For example, the large config uses `gradient_accumulation_steps: 40`, so use `10` on 4 GPUs for roughly the same tokens per optimizer update.

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
