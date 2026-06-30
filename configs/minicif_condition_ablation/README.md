# Minicif Conditioning Ablations

This folder contains small, regularized configs for comparing PXRD conditioning choices.

## 1. Set Up The Environment

From the repository root:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install torch numpy pandas matplotlib seaborn pyYAML tqdm omegaconf h5py pymatgen periodictable scikit-learn
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

## 2. Check The Dataset

The configs expect this dataset layout:

```text
data/noma/serialized/train.h5
data/noma/serialized/val.h5
data/noma/serialized/test.h5
```

Check it with:

```bash
ls -lh data/noma/serialized/train.h5 \
       data/noma/serialized/val.h5 \
       data/noma/serialized/test.h5
```

If the data is not prepared yet, prepare it first:

```bash
python bin/prepare_minicif_dataset.py --help
```

Use the command matching your raw input files. The important output is the three HDF5 files above.

## 3. What Each Config Tests

```text
small_no_condition.yaml       no PXRD conditioning baseline
small_mlp_insert.yaml         old dense PXRD MLP conditioning, inserted tokens
small_conv_insert.yaml        dense PXRD convolution encoder, inserted tokens
small_peak_insert.yaml        sparse peak-list encoder, inserted tokens
small_hybrid_insert.yaml      dense PXRD + peak-list encoder, inserted tokens
small_conv_cross.yaml         dense PXRD convolution encoder, cross-attention
small_hybrid_cross.yaml       dense PXRD + peak-list encoder, cross-attention
```

All configs use clean PXRD settings: no noise, no peak dropout, no q-shift, no impurity peaks, and no background augmentation.

## 4. Sanity Check Before Launching

Print the commands that would run:

```bash
python bin/run_minicif_condition_ablation.py --dry-run
```

Run one config manually if you want a single-job smoke test:

```bash
python bin/train.py --config configs/minicif_condition_ablation/small_hybrid_cross.yaml
```

For a smaller temporary smoke run, edit a copy of a config and reduce:

```yaml
max_iters: 10
eval_interval: 5
eval_iters_train: 2
eval_iters_val: 2
batch_size: 8
batch_token_budget: 2048
```

## 5. Run All Ablations Sequentially

On the cluster:

```bash
sbatch minislurm/run_minicif_condition_ablation.sh
```

This runs every YAML in this folder, one after the other, on one GPU.

Run only selected configs:

```bash
sbatch minislurm/run_minicif_condition_ablation.sh \
  --configs configs/minicif_condition_ablation/small_conv_cross.yaml \
            configs/minicif_condition_ablation/small_hybrid_cross.yaml
```

Run locally in the current shell:

```bash
python bin/run_minicif_condition_ablation.py
```

## 6. Watch Progress

Training logs go to each config's `out_dir`, for example:

```text
minicif_ablation_small_hybrid_cross/
```

Check the latest metrics with:

```bash
tail -n 20 minicif_ablation_small_hybrid_cross/metrics.jsonl
```

Check Slurm output with:

```bash
tail -f logs/minicif_ablation_<jobid>.out
```

## 7. Evaluate A Trained Model

After training, evaluate a checkpoint:

```bash
python bin/visualize_minicif.py \
  --checkpoint minicif_ablation_small_hybrid_cross/ckpt.pt \
  --dataset-dir data/noma \
  --out-dir minicif_ablation_small_hybrid_cross/report \
  --splits val test \
  --max-items 200 \
  --num-reps 8 \
  --generation-batch-size 8 \
  --prompt-modes pxrd pxrd-elements pxrd-elements-cs pxrd-elements-cs-sg \
  --top-k 50
```

The report writes:

```text
minicif_ablation_small_hybrid_cross/report/minicif_summary.csv
minicif_ablation_small_hybrid_cross/report/minicif_generation_metrics.csv
minicif_ablation_small_hybrid_cross/report/learning_curves.png
minicif_ablation_small_hybrid_cross/report/metric_summary.png
```

For a quick evaluation smoke test:

```bash
python bin/visualize_minicif.py \
  --checkpoint minicif_ablation_small_hybrid_cross/ckpt.pt \
  --dataset-dir data/noma \
  --out-dir minicif_ablation_small_hybrid_cross/report_smoke \
  --splits val \
  --max-items 10 \
  --num-reps 2 \
  --generation-batch-size 2 \
  --prompt-modes pxrd pxrd-elements
```

## 8. Compare Results

After evaluating several models, compare the summaries:

```bash
python - <<'PY'
import glob
import os
import pandas as pd

frames = []
for path in glob.glob("minicif_ablation_*/report/minicif_summary.csv"):
    df = pd.read_csv(path)
    df.insert(0, "run", os.path.dirname(os.path.dirname(path)))
    frames.append(df)

summary = pd.concat(frames, ignore_index=True)
cols = [
    "run",
    "split",
    "prompt_mode",
    "valid_minicif_rate",
    "finish_rate",
    "best_of_k_match_rate",
    "element_set_accuracy",
    "mean_extra_elements",
    "mean_missing_elements",
    "median_best_rwp",
]
print(summary[cols].sort_values(["split", "prompt_mode", "best_of_k_match_rate"], ascending=[True, True, False]).to_string(index=False))
PY
```

## 9. Interpret The Main Signals

Use these first:

```text
valid_minicif_rate       Are generations syntactically parseable?
finish_rate              Does the model emit </mcif> before max_new_tokens?
element_set_accuracy     Does pxrd-elements prompting keep the requested elements?
mean_extra_elements      How often does the model add extra elements?
mean_missing_elements    How often does it drop requested elements?
best_of_k_match_rate     Does any candidate match the reference?
median_best_rwp          Does the generated structure reproduce the PXRD?
```

The most useful comparisons are:

```text
small_no_condition vs small_conv_insert
small_conv_insert vs small_conv_cross
small_peak_insert vs small_conv_insert
small_hybrid_insert vs small_hybrid_cross
small_hybrid_cross vs small_conv_cross
```

If `pxrd-elements` has high `mean_extra_elements`, use constrained decoding and shorter `max_new_tokens` first, then compare whether cross-attention or hybrid conditioning improves it.
