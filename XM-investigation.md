# Explorative Modeling for deCIFer

## Status

Forward Explorative Modeling (XM) is available as an experimental training
option. The model architecture and existing hyperparameters remain unchanged
when `xm_best_of_k: 1`.

The implementation was based on the [XM paper](https://arxiv.org/abs/2607.27372),
the [official implementation](https://github.com/alexiglad/XM), and the
[project page](https://explorative-modeling.github.io/). The strongest reported
results concern image, video, and masked-diffusion models. The paper describes
autoregressive language-model evidence as preliminary, so deCIFer needs a
controlled application-specific comparison.

## Mapping to deCIFer

| XM concept | deCIFer value |
| --- | --- |
| Fixed target | One tokenized miniCIF structure |
| Fixed conditioning | PXRD, constituents, stoichiometry, Wyckoff data, lattice constraints, artifact realization, and condition-drop decision |
| Explored stochastic variable | A learned discrete XM mode embedding prepended to the sequence |
| Candidate ranking loss | Mean next-token cross-entropy for each structure |
| Final batch loss | Winning per-structure losses weighted by valid token count |

XM requires record-aligned batches because candidate selection needs one target
per batch row. Packed batches can contain several structures in one row and are
rejected when `xm_best_of_k > 1`.

For a batch size `B`, target length `T`, and exploration count `K`:

```text
input tokens          [B, T]
candidate mode IDs    [K, B]
candidate losses      [K, B]
winner candidate IDs  [B]
winner mode IDs       [B]
```

Candidate selection is `candidate_losses.min(dim=0)`. This chooses a winner
independently for every structure. The winning modes are forwarded once more
with gradient tracking, so gradients reach only each structure's winning path.

## Training behavior

`K = 1` executes the original model call directly. It creates no XM embedding,
draws no extra random values, and produces bitwise-identical logits and loss in
the regression test.

For `K > 1`, training uses the memory-saving XM algorithm:

1. Sample `K` mode IDs per structure.
2. Evaluate all candidates without gradient tracking.
3. Select the lowest-loss candidate independently per structure.
4. Re-evaluate the winners with gradient tracking and backpropagate once.

The CPU and CUDA random-number states are restored for each candidate. Dropout,
classifier-free condition dropout, and stochastic artifact inputs are therefore
shared. The mode embedding is the only explored variable.

Validation uses one randomly sampled XM mode and the original token-level loss.
This keeps validation loss comparable across `K`. Training logs contain the XM
winner loss, which is expected to decrease mechanically as `K` grows.

Generation samples one mode per generated output and keeps it fixed for the
whole autoregressive sequence. Inference still uses one model pass per token and
adds one prefix token plus `K * n_embd` learned parameters. Multiple requested
repetitions can sample different modes through the existing generation
workflow.

## Compute and memory

The memory-saving implementation runs `K` forward passes without retained
autograd activations, one forward pass with activations, and one backward pass.
Peak activation memory is close to the baseline. For a Transformer training
step approximated as one forward plus one backward, the expected compute
factors are:

| K | Approximate training compute |
| ---: | ---: |
| 1 | 1.00x |
| 2 | 1.67x |
| 4 | 2.33x |
| 8 | 3.67x |

A small CPU mechanics benchmark on PyTorch 2.10 used one CPU thread, AdamW,
batch size 2, length 16, and a two-layer 64-wide model. It measures
implementation overhead, not model quality:

| K | Step time | Data tokens/s |
| ---: | ---: | ---: |
| 1 | 3.72 ms | 8,601 |
| 2 | 8.17 ms | 3,918 |
| 4 | 10.07 ms | 3,177 |
| 8 | 15.09 ms | 2,121 |

Run the same benchmark on the training GPU to record representative wall time
and peak allocated memory:

```bash
python bin/benchmark_xm.py \
  --device cuda \
  --dtype bfloat16 \
  --batch-size 8 \
  --sequence-length 768 \
  --layers 8 \
  --heads 8 \
  --embedding-dim 512 \
  --steps 50 \
  --output xm_gpu_benchmark.json
```

## Controlled experiment

Submit equal-update runs:

```bash
bash minislurm/run_xm_sweep.sh equal-updates
```

Submit approximately equal-compute runs, where the update count is scaled by
`3 / (K + 3)` for `K > 1`:

```bash
bash minislurm/run_xm_sweep.sh equal-compute
```

Override the base model, update budget, or output root with environment
variables:

```bash
REPRESENTATION=hybrid \
SIZE=medium \
BASE_ITERS=100000 \
OUTPUT_ROOT=models/minicif_v2/xm/hybrid/standard/sweeps \
bash minislurm/run_xm_sweep.sh equal-updates
```

The dedicated training configs are located under
`configs/config_v2/xm/{peak,dense,hybrid}/standard/{k2,k4,k8}/`. K=1 uses the
corresponding existing standard config.

Evaluate a completed run through the existing resumable evaluator:

```bash
CHECKPOINT=models/minicif_v2/xm/peak/standard/sweeps/equal-updates/k2/medium/ckpt.pt \
OUT_DIR=models/minicif_v2/xm/peak/standard/sweeps/equal-updates/k2/medium/minicif_report \
sbatch minislurm/evaluate_minicif_v2.sh \
  --splits test \
  --prompt-modes pxrd-elements \
  --num-reps 8 \
  --refinement-config configs/refinement/quick.yaml
```

Evaluate every run with the same split, prompt modes, number of repetitions,
artifact seed, and refinement preset. Compare:

- validation loss and convergence wall time;
- valid miniCIF and structure rates;
- initial and refined Rwp distributions, globally and by crystal system;
- RMSD, composition, space-group, and crystal-system accuracy;
- unique miniCIF, formula, space-group, and crystal-system counts per input;
- GPU peak memory and total training time;
- equal-update and equal-compute results separately.

The evaluation summary now records `mean_unique_minicif_fraction`,
`mean_unique_formulas_per_sample`, `mean_unique_space_groups_per_sample`, and
`mean_unique_crystal_systems_per_sample` for coverage checks.

## Recommendation

Start with `K = 2` on the peak standard medium model in both comparison
settings. Continue to `K = 4` when Rwp, structure recovery, and coverage improve
together. `K = 8` carries a large training cost and should follow evidence from
the smaller runs.

XM has the clearest potential for ambiguous prompts such as PXRD-only or
constituents-only generation, where several structures can be compatible with
one observation. A strongly conditioned PXRD pattern may leave less useful
multimodality for the learned modes to separate. Validation and refined fit
quality should drive the decision.

Reverse XM is intentionally absent from this baseline. deCIFer does not yet
define a defensible sampler for multiple compatible target structures, and
minimum-over-target training needs explicit coverage and mode-collapse controls.
