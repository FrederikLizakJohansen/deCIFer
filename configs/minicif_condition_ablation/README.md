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
For `peak` and `hybrid` configs, `qmin` and `qmax` also define the fixed q range used to normalize peak-list q positions.

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

## 8. Evaluate Checkpoints On The Cluster

Use the existing SLURM evaluation wrapper for a single checkpoint:

```bash
sbatch minislurm/evaluate_minicif.sh \
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

Submit one evaluation job per ablation checkpoint:

```bash
for run in \
  minicif_ablation_small_no_condition \
  minicif_ablation_small_mlp_insert \
  minicif_ablation_small_conv_insert \
  minicif_ablation_small_peak_insert \
  minicif_ablation_small_hybrid_insert \
  minicif_ablation_small_conv_cross \
  minicif_ablation_small_hybrid_cross
do
  sbatch minislurm/evaluate_minicif.sh \
    --checkpoint "${run}/ckpt.pt" \
    --dataset-dir data/noma \
    --out-dir "${run}/report" \
    --splits val test \
    --max-items 200 \
    --num-reps 8 \
    --generation-batch-size 8 \
    --prompt-modes pxrd pxrd-elements pxrd-elements-cs pxrd-elements-cs-sg \
    --top-k 50
done
```

For a quick cluster smoke test before launching full evaluations:

```bash
sbatch minislurm/evaluate_minicif.sh \
  --checkpoint minicif_ablation_small_hybrid_cross/ckpt.pt \
  --dataset-dir data/noma \
  --out-dir minicif_ablation_small_hybrid_cross/report_smoke \
  --splits val \
  --max-items 10 \
  --num-reps 2 \
  --generation-batch-size 2 \
  --prompt-modes pxrd pxrd-elements
```

SLURM logs are written here:

```text
logs/minicif_eval_<jobid>_0.out
```

The evaluation report for each run is written to its `--out-dir`.

## 9. Compare Results

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

## 10. Interpret The Main Signals

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

## 11. Next Experiments After This Ablation

Use the first ablation results to choose the next config family.

If cross-attention helps:

```text
Try q-aware dense patch tokens instead of adaptive pooled conv tokens.
Try cross-attention every 2 layers instead of every layer.
Try hybrid cross-attention with fewer peak tokens and more dense tokens.
```

If peak-list or hybrid conditioning helps:

```text
Add peak width, prominence, and confidence channels.
Add a real-PXRD preprocessing path that detects peaks from measured traces.
Keep dense trace + detected peaks together for real PXRD; do not rely on peak picking alone.
```

If `pxrd-elements` still adds wrong elements:

```text
Add explicit element-set or formula conditioning tokens.
Train with PXRD-only, PXRD+elements, PXRD+formula, and PXRD+formula+crystal-system variants.
Use component dropout so the model can run both with and without known components.
```

If shifted or real patterns perform poorly:

```text
Add PXRD pre-alignment experiments.
Compare raw PXRD, aligned PXRD, PXRD+components, and aligned PXRD+components.
Evaluate on synthetic q-shift/q-scale sweeps before trusting real-pattern gains.
```

The detailed backlog for these ideas is in:

```text
minicif-idea-for-improvements.md
```

The next implementation target is contrastive PXRD encoder pretraining:

```text
1. Pretrain the condition encoder on two augmented PXRD views of the same structure.
2. Save the pretrained condition encoder checkpoint.
3. Initialize minicif training from that encoder.
4. Compare against the same config trained from scratch.
```

Run the default hybrid encoder pretraining job:

```bash
sbatch minislurm/pretrain_pxrd_encoder.sh
```

Or run it locally:

```bash
PYTHONPATH=. python bin/pretrain_pxrd_encoder.py \
  --config configs/minicif_pxrd_encoder_pretrain_hybrid.yaml
```

For a CUDA laptop smoke test, first run the synthetic probe:

```bash
PYTHONPATH=. python bin/pretrain_pxrd_encoder.py \
  --config configs/minicif_pxrd_encoder_pretrain_synthetic_debug.yaml
```

This does not import or read HDF5 data. If it segfaults, debug the CUDA/PyTorch/model path first: try `device: 'cpu'`, lower `batch_size`, and keep `dtype: 'float32'`.

Then run the real-data smoke config:

```bash
PYTHONPATH=. python bin/pretrain_pxrd_encoder.py \
  --config configs/minicif_pxrd_encoder_pretrain_hybrid_laptop_smoke.yaml
```

This uses:

```yaml
num_workers_dataloader: 0
pin_memory: False
preload_dataset_to_memory: False
dtype: 'float32'
live_plot: False
```

If the synthetic probe works but this config segfaults, inspect the local HDF5/h5py stack or the serialized data file. As an opt-in diagnostic fallback, set `preload_dataset_to_memory: True`; that reads `train.h5` into CPU memory once, closes the HDF5 file, and only then enters the CUDA training loop.

If that works, turn options back on one at a time:

```yaml
live_plot: True
dtype: 'float16'
num_workers_dataloader: 2
dataloader_multiprocessing_context: 'spawn'
```

For the full run, keep `dataloader_multiprocessing_context: 'spawn'` when `num_workers_dataloader > 0`. HDF5 plus PyTorch worker multiprocessing can crash at the C-library level on some Linux/CUDA laptop setups.

Watch these files while it trains:

```text
minicif_pxrd_encoder_pretrain_hybrid/contrastive_live.png
minicif_pxrd_encoder_pretrain_hybrid/latest_metrics.json
minicif_pxrd_encoder_pretrain_hybrid/contrastive_metrics.csv
```

The live plot updates every `plot_interval` iterations. Good signs are decreasing loss, positive similarity rising above negative similarity, and retrieval top-1 increasing above the random baseline.

Then copy one of the matching hybrid configs and add:

```yaml
pretrained_condition_encoder_path: 'minicif_pxrd_encoder_pretrain_hybrid/pxrd_encoder_pretrain.pt'
freeze_pretrained_condition_encoder: False
```

For an encoder-only comparison, set `freeze_pretrained_condition_encoder: True`.

### Analyze a Finished PXRD Encoder

After pretraining finishes, create publication-style diagnostics for the encoder:

```bash
PYTHONPATH=. python bin/analyze_pxrd_encoder.py \
  --checkpoint minicif_pxrd_encoder_pretrain_hybrid/pxrd_encoder_pretrain.pt \
  --dataset-dir data/noma \
  --split val \
  --max-samples 2000 \
  --batch-size 64 \
  --tsne
```

The script writes an `encoder_analysis_val/` folder beside the checkpoint. Important outputs:

```text
analysis_summary.json
figure_manifest.json
training_metrics.png/.pdf
pca2_by_crystal_system.png/.pdf
tsne2_by_crystal_system.png/.pdf
pca_explained_variance.png/.pdf
embedding_similarity_heatmap.png/.pdf
pxrd_similarity_vs_embedding_similarity.png/.pdf
augmentation_invariance.png/.pdf
sample_pxrd_traces.png/.pdf
encoder_embeddings.npz
sample_summary.csv
hard_negatives_by_crystal_system.csv
representative_samples_by_crystal_system.csv
```

Use the summary JSON for numbers in tables: final contrastive metrics, k-nearest-neighbor label agreement, silhouette scores, low-dimensional trustworthiness, PXRD-similarity correlation, and augmentation-invariance margins. Use the hard-negative CSV to inspect structures that the encoder considers similar even though their labels differ.
