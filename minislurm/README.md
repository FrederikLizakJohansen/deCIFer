# Minicif SLURM Scripts

Run from the repository root after activating the Python environment.

## Minicif v2 quick start

The v2 workflow uses a separate dataset and checkpoint namespace, so it does not
overwrite legacy minicif artifacts.

```bash
sbatch minislurm/prepare_minicif_v2_dataset.sh
sbatch minislurm/train_minicif_v2.sh
sbatch minislurm/evaluate_minicif_v2.sh
```

Dataset preparation defaults to `--xrd-backend auto`, which prefers
BraggCalculator when installed and otherwise uses pymatgen. To guarantee the
fast backend is active:

```bash
python -m pip install -e .
sbatch minislurm/prepare_minicif_v2_dataset.sh \
  --xrd-backend braggcalculator
```

The repository supports Python 3.12 and 3.13, as required by BraggCalculator
0.3.0. Dataset preparation excludes structural targets longer than 769 tokens
before diffraction calculation; pass `--max-token-length 0` only for an
intentional larger-context dataset.

Defaults are `data/noma` for the raw gzip source,
`data/noma_minicif_v2` for prepared data, and
`configs/minicif_v2_medium_config.yaml` for training. Override them through
environment variables:

```bash
RAW_DIR=data/noma OUT_DIR=data/noma_minicif_v2 \
  sbatch minislurm/prepare_minicif_v2_dataset.sh

CONFIG=configs/minicif_v2_small_config.yaml \
  sbatch minislurm/train_minicif_v2.sh

CHECKPOINT=minicif_v2_model_small/ckpt.pt \
DATASET_DIR=data/noma_minicif_v2 \
OUT_DIR=minicif_v2_model_small/minicif_report \
  sbatch minislurm/evaluate_minicif_v2.sh
```

Before the first full job, run the short CUDA integration config:

```bash
sbatch minislurm/train_minicif_v2.sh \
  --config configs/minicif_v2_hybrid_artifacts_smoke.yaml
```

This performs only a few optimizer steps and is not intended to produce a useful
checkpoint.

The full BraggCalculator artifact model uses clean sparse HDF5 records and
samples artifacts during training:

```bash
sbatch minislurm/train_minicif_v2.sh \
  --config configs/minicif_v2_medium_hybrid_artifacts.yaml
```

For deterministic artifact evaluation:

```bash
CHECKPOINT=minicif_v2_model_medium_hybrid_artifacts/ckpt.pt \
OUT_DIR=minicif_v2_model_medium_hybrid_artifacts/minicif_report \
sbatch minislurm/evaluate_minicif_v2.sh \
  --artifact-config configs/xrd_artifacts/full_evaluation.yaml
```

The training artifact profile is intentionally unseeded. The evaluation profile
has a fixed seed. Omit `--artifact-config` for clean conditions.

`train_minicif_v2.sh` runs `bin/audit_minicif_v2.py` before allocating model
memory. Set `AUDIT_MAX_ITEMS=0` to deeply validate every record, or
`SKIP_PREFLIGHT=1` only after an unchanged dataset has already passed.

Run the preflight directly on a login/CPU node:

```bash
python bin/audit_minicif_v2.py \
  --config configs/minicif_v2_medium_config.yaml \
  --max-items 100 \
  --output minicif_v2_preflight.json
```

Optional Fourier peak-encoder pretraining uses:

```bash
sbatch minislurm/pretrain_pxrd_encoder.sh \
  --config configs/minicif_v2_pxrd_encoder_pretrain.yaml
```

Set `pretrained_condition_encoder_path` in a v2 training config to the resulting
`minicif_v2_pxrd_encoder_pretrain/pxrd_encoder_pretrain.pt` only when the encoder
width and Fourier-token settings match.

Hybrid encoder pretraining with the full artifact profile uses:

```bash
sbatch minislurm/pretrain_pxrd_encoder.sh \
  --config configs/minicif_pxrd_encoder_pretrain_bragg_artifacts.yaml
```

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

Analyze a finished PXRD encoder checkpoint:

```bash
PYTHONPATH=. python bin/analyze_pxrd_encoder.py \
  --checkpoint minicif_pxrd_encoder_pretrain_hybrid/pxrd_encoder_pretrain.pt \
  --dataset-dir data/noma \
  --split val \
  --max-samples 2000 \
  --batch-size 64 \
  --tsne
```

This writes publication-style diagnostics, CSVs, and PDFs to `minicif_pxrd_encoder_pretrain_hybrid/encoder_analysis_val`.

Additional encoder-pretraining ablations:

```bash
sbatch minislurm/pretrain_pxrd_encoder.sh \
  --config configs/minicif_pxrd_encoder_pretrain_hybrid_pooled.yaml

sbatch minislurm/pretrain_pxrd_encoder.sh \
  --config configs/minicif_pxrd_encoder_pretrain_hybrid_pxrd_metric.yaml

sbatch minislurm/pretrain_pxrd_encoder.sh \
  --config configs/minicif_pxrd_encoder_pretrain_hybrid_aux_metric.yaml
```

Analyze each output directory and compare the `embedding_spaces.pooled` metrics in `analysis_summary.json`.

The pretraining configs resume automatically. If a run is stopped by `Ctrl-C` locally or `scancel <job_id>` on SLURM, the script saves `pxrd_encoder_pretrain.pt` at the next safe point. Rerun the same command to continue. To force a fresh run, set `resume: False` or remove the output directory.

If a job appears stuck in data loading, set `num_workers_dataloader: 0`. If some iterations are just slow, reduce `max_raw_peaks_per_sample` from `2048` to `1024`.

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
