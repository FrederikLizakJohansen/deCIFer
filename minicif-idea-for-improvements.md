# minicif ideas for improvements

Working name: `minicif`

Purpose: collect concrete ideas for improving the current deCIFer codebase, model architecture, training regime, data augmentation, evaluation, and generation workflow. Keep entries structured enough that each can later become a small experiment, issue, or commit.

## Progress log

- 2026-06-27: Started notes file and reviewed `bin/train.py` for immediate training-code improvements. No source code changes made.
- 2026-06-27: Implemented the immediate training-code improvements in `bin/train.py` and `decifer/decifer_model.py`; verified with syntax checks, an attention opt-in forward check, and a synthetic CPU training smoke test.
- 2026-06-27: Started aggressive CIF minimization with a standalone `decifer.minicif` canonicalizer and focused unit tests.
- 2026-06-27: Added a one-pass minicif dataset preparation path and moved PXRD conditioning toward sparse peak storage plus training-time Nyquist-aware augmentation.
- 2026-06-27: Added first-pass minicif constrained decoding helpers: `sg_*` choices are masked by the emitted crystal system, and `<atom>` element choices are masked to the constituent elements emitted in the minicif prefix.
- 2026-06-28: Added a configurable PXRD condition encoder. The old one-vector MLP remains available, and minicif can now use a small 1D convolutional PXRD encoder that emits multiple latent condition tokens per `<mcif>` start.
- 2026-06-28: Verified conditioned attention masking with a regression test for packed minicif records, added minicif-to-structure rendering, and added `bin/visualize_minicif.py` for learning curves plus validation/test match-rate and Rwp reports.
- 2026-06-28: Added structured training metrics logs (`metrics.jsonl` and `metrics.csv`) and created `minicif-features-vs-decifer.md` to summarize core minicif changes relative to deCIFer.

## Review assumptions

- This file is an idea backlog, not a commitment to implement everything.
- Immediate items below are based on a code read of `bin/train.py` and the model interface in `decifer/decifer_model.py`.
- Priority means expected value for stabilizing or accelerating experiments, not necessarily scientific novelty.

## Immediate `bin/train.py` improvements

### P0 - Fix gradient accumulation scaling

Status: implemented 2026-06-27.

Observation: the training loop accumulates `gradient_accumulation_steps` losses, but each micro-step backpropagates the full loss.

Relevant code: `bin/train.py`, around the micro-step loop where `scaler.scale(loss).backward()` is called.

Why it matters: changing `gradient_accumulation_steps` changes the effective gradient scale, so training behavior is coupled to accumulation settings. This makes learning-rate comparisons and batch-size sweeps harder to interpret.

Proposed experiment/fix: divide `loss` by `gradient_accumulation_steps` before backpropagation, and log the unscaled loss separately if desired.

Verification:
- Run a tiny smoke training config before/after.
- Confirm one optimizer step with `N` accumulation steps matches the gradient scale of a manually averaged `N`-microbatch loss.

### P0 - Avoid storing attention maps during every training forward pass

Status: implemented 2026-06-27. Attention capture is now opt-in through `return_attn`/`plot_attention`.

Observation: `Decifer.forward()` always calls each block with `return_attn=True` and stores detached CPU attention tensors in `self.attn_scores`.

Relevant code: `decifer/decifer_model.py`, forward pass through transformer blocks.

Why it matters: this likely adds avoidable memory movement, CPU allocation, and attention materialization overhead during normal training. It may also prevent the faster flash-attention path because attention weights are requested.

Proposed experiment/fix: make attention capture optional, default off during training, and only enable it for interpretability/debug runs.

Verification:
- Compare tokens/sec or iteration time before/after on the same small config.
- Confirm attention visualization workflows can still request attention explicitly.

### P1 - Clean up duplicated and unused config fields

Status: implemented 2026-06-27. Kept one `block_size`, removed unused `cond_size`/training `wavelength`, and wired `always_save_checkpoint`.

Observation: `TrainConfig` defines `block_size` twice; the second definition silently wins. `cond_size`, `always_save_checkpoint`, and `wavelength` appear unused in the current training path.

Relevant code: `bin/train.py`, `TrainConfig`.

Why it matters: silent config shadowing and unused fields make experiments ambiguous. For example, a default of `2048` appears in the data section, but the effective default is `1024`.

Proposed experiment/fix:
- Keep one `block_size` field.
- Remove or wire up unused fields.
- If `always_save_checkpoint` is intended, make checkpointing honor it.

Verification:
- Print/parse the structured config and confirm defaults are unambiguous.
- Run `python bin/train.py --config configs/deCIFer_NOMA_small_config.yaml` far enough to validate config parsing.

### P1 - Make checkpoint behavior match configuration

Status: implemented 2026-06-27.

Observation: checkpoints are saved only during validation and only after iteration 0. If `validate: False`, the loop does not save checkpoints. `always_save_checkpoint` is defined but not used.

Relevant code: `bin/train.py`, evaluation/checkpoint block.

Why it matters: long training runs can lose progress if validation is disabled or if a job stops before the first validation checkpoint.

Proposed experiment/fix:
- Separate periodic "current state" checkpointing from "best validation model" checkpointing.
- Save on `always_save_checkpoint` even without validation.
- Consider saving an initial config/checkpoint manifest at run start.

Verification:
- Run a tiny config with `validate: False` and `always_save_checkpoint: True`; confirm `ckpt.pt` is written.
- Resume from that checkpoint and confirm iteration count and optimizer state continue.

### P1 - Make validation deterministic and augmentation-aware

Status: implemented 2026-06-27. Validation/test loading is sequential, validation resets its iterator, and evaluation uses clean deterministic XRD conversion.

Observation: `estimate_loss()` uses the same `get_batch()` path as training, including random loader iteration and XRD augmentation when conditioning is enabled.

Relevant code: `bin/train.py`, `estimate_loss()` and `get_batch()`.

Why it matters: validation loss can include augmentation noise and random packing effects. That makes early stopping and best-model selection noisier.

Proposed experiment/fix:
- Add an `augment` flag to batch creation.
- Disable stochastic augmentation for validation by default.
- Optionally report both clean validation loss and augmented robustness loss.

Verification:
- Repeated validation calls at the same checkpoint should be much less noisy when clean validation is selected.

### P1 - Handle conditioning alignment explicitly when packing CIFs

Status: partially implemented 2026-06-27. Added optional `debug_batch_assertions`; a fuller synthetic unit test for sequence-to-condition mapping is still worth adding.

Observation: `get_batch()` concatenates multiple CIF token sequences into packed blocks, then gathers conditioning vectors based on cumulative sequence lengths and start-token positions.

Relevant code: `bin/train.py`, `get_batch()` sequence packing and `cond_batch` slicing.

Why it matters: this is a fragile area. If a packed block truncates a CIF or includes different counts of `START_ID` markers than expected, the conditioning vectors and inserted condition embeddings can become misaligned.

Proposed experiment/fix:
- Add lightweight assertions in debug mode: number of inserted condition vectors equals number of start markers used by the batch.
- Consider returning explicit sequence-to-condition mapping from the packing step.

Verification:
- Unit test a synthetic batch with known sequence lengths, truncation, and multiple packed blocks.
- Confirm condition vectors align with the intended CIF starts.

### P2 - Improve data loading reproducibility and performance knobs

Status: implemented 2026-06-27.

Observation: seeding only calls `torch.manual_seed`. Python `random`, NumPy, worker seeds, and `DataLoader` generator state are not set. `pin_memory()` is called even on the non-CUDA branch.

Relevant code: `bin/train.py`, seed setup and `get_batch()`.

Why it matters: exact reruns are harder, and CPU/MPS runs may pay unnecessary overhead or fail depending on device behavior.

Proposed experiment/fix:
- Seed `random` and `numpy`.
- Pass a seeded `torch.Generator` into samplers/loaders.
- Add worker seed initialization if `num_workers_dataloader > 0`.
- Only use pinned memory for CUDA transfers.

Verification:
- Two short runs with the same seed should produce matching first batches and early losses, subject to CUDA nondeterminism settings.

### P2 - Modernize mixed precision device handling

Status: implemented 2026-06-27 for CUDA-vs-non-CUDA handling. MPS-specific training remains untested.

Observation: AMP and `GradScaler` use `torch.cuda.amp` whenever `device != "cpu"`, which does not distinguish CUDA from MPS or other devices.

Relevant code: `bin/train.py`, AMP context and scaler setup.

Why it matters: this can break or silently behave oddly on non-CUDA devices.

Proposed experiment/fix: choose AMP/scaler based on the actual device type, for example CUDA-only for `float16` scaling unless other devices are explicitly supported.

Verification:
- Run config parsing and one forward/backward step on CPU.
- If MPS support is desired, run a specific MPS smoke test.

### P2 - Save richer run metadata

Status: implemented 2026-06-27.

Observation: checkpoints include config and metrics, but not obvious run metadata such as git commit, command, wall-clock timing, parameter count, dataset paths resolved at runtime, or environment versions.

Why it matters: minicif experiments will likely involve many small architecture/training/data changes. Reproducibility will depend on knowing exactly what produced each checkpoint.

Proposed experiment/fix: write a compact `run_metadata.yaml` or include metadata in the checkpoint.

Verification:
- Start a run and confirm metadata can identify the code version, config, command, and dataset root.

## Idea backlog

### Model architecture

- P0 - Replace single-vector PXRD injection with cross-attention over learned PXRD tokens.
  Current conditioning compresses the full continuous PXRD vector through an MLP and inserts one condition embedding at each CIF start. That is a strong bottleneck: local peak positions, widths, missing peaks, and uncertainty all have to fit into one vector. A stronger minicif architecture would encode the PXRD as a sequence of q/intensity patches or peaks, then let CIF tokens cross-attend to that representation.
  Experiment: implement a small PXRD encoder producing 32-128 latent tokens; add cross-attention blocks every N transformer layers or prefix the latents as non-generated memory tokens. Compare validation loss, generated structure validity, and PXRD agreement at fixed parameter count.

  Current implementation:
  - Added `condition_encoder: mlp|conv`.
  - Added `condition_n_tokens`, allowing either encoder to emit multiple non-generated condition tokens per `<mcif>` start.
  - Added a compact 1D convolutional PXRD encoder with adaptive pooling over q-space, projection to transformer width, and learned latent-token positions.
  - Kept condition insertion aligned with packed batches by inserting the latent tokens at each start token, rather than adding one prefix for the whole row.

  Still needed:
  - True cross-attention from CIF tokens into PXRD memory tokens.
  - Peak-list or patch-aware encoders that preserve explicit q coordinates rather than only dense-grid intensity order.

- P0 - Add composition and lattice priors as explicit conditioning channels.
  PXRD alone is ambiguous, and CIF generation has hard chemistry/geometric constraints. If composition, formula, crystal system, or approximate cell parameters are available at generation time, encode them separately instead of expecting the language model to infer everything from diffraction.
  Experiment: train variants with PXRD-only, PXRD+composition, PXRD+space-group/crystal-system, and PXRD+cell hints. Measure validity and ambiguity reduction on held-out structures.

- P1 - Use a structure-aware decoder head or constrained field heads for numeric CIF values.
  Autoregressive text loss treats CIF numbers as character/token strings. Lattice constants, angles, fractional coordinates, occupancies, and symmetry labels have different semantics. Minicif could keep text generation for syntax but add auxiliary heads for key numeric fields, or generate an intermediate structured representation before formatting CIF.
  Experiment: predict lattice parameters and space group from the hidden state at `data_`, add auxiliary losses, and check whether generated CIFs become more physically plausible.

- P1 - Investigate modern Transformer blocks.
  The current model is close to a nanoGPT-style block. Candidate low-risk upgrades: RMSNorm, SwiGLU/GEGLU MLPs, RoPE or ALiBi positions, residual scaling, and flash-attention-friendly masking.
  Experiment: keep parameter count and training budget fixed, swap one architectural change at a time, and track tokens/sec plus clean validation loss.

- P2 - Hierarchical generation: scaffold first, details second.
  CIFs have natural hierarchy: formula/space group/cell, symmetry ops, atom sites. A two-stage model could first generate a compact structure plan, then generate the full CIF conditioned on it.
  Experiment: derive plans from existing CIFs automatically and train plan-to-CIF generation; compare invalid-CIF rate and PXRD match.

### Training regime

- P0 - Establish reproducible baselines before changing model capacity.
  Now that checkpointing, validation, and gradient accumulation are saner, run the current small and full configs to establish loss, validity, PXRD agreement, throughput, and memory. Without this, architecture changes will be hard to interpret.

  Baseline protocol:
  - Use `configs/minicif_small_config.yaml` plus `minislurm/train_minicif.sh` as the first smoke/baseline path.
  - Record train loss, clean validation loss, tokens/sec, max GPU memory, checkpoint git commit, dataset path, q-grid size, and augmentation settings.
  - Generate a small fixed validation sample set with and without `minicif_constrained_decoding` so syntax/chemistry gains are separated from training-loss changes.
  - Treat later architecture changes as meaningful only if they beat this run at comparable tokens seen and parameter count.

  Current implementation:
  - Training now writes `metrics.jsonl` and `metrics.csv` in the run directory.
  - Metrics include train/eval event type, iteration, learning rate, train/validation loss, step loss, step time, token throughput, gradient norm, max GPU memory, q-step, tokenizer, condition encoder, condition token count, batch size, block size, and accumulation steps.

- P0 - Add curriculum over PXRD difficulty.
  Start with clean, fixed-width simulated patterns and gradually introduce peak broadening, noise, missing regions, intensity scaling, preferred-orientation-like perturbations, and background. This can make the conditioning problem easier early and more robust later.
  Experiment: schedule augmentation ranges over training iterations and compare to full-strength augmentation from step 0.

- P1 - Use token-budgeted batching instead of fixed sequence-count batching.
  Current packing makes each optimizer step depend on how many full blocks happen to fit after concatenation. A token-budgeted batcher would make training more predictable and improve hardware utilization.
  Experiment: build batches by target token count, track effective tokens/step, and compare loss curves normalized by tokens seen.

- P1 - Add auxiliary denoising/contrastive objectives for PXRD conditioning.
  The model should learn that different noisy views of the same structure have the same underlying CIF. Add an auxiliary contrastive loss between PXRD embeddings from two augmentations of the same sample, or predict clean PXRD features from augmented inputs.

- P2 - Try EMA checkpoints for generation.
  An exponential moving average of weights often improves sample quality even when validation loss is similar.
  Experiment: maintain EMA weights and evaluate generated CIF validity/PXRD agreement from current vs EMA checkpoints.

### Augmented data

- P0 - Make the PXRD augmentation model more physically realistic.
  Current augmentation covers broadening, noise, intensity scale, and masking. Real experimental patterns also include background, zero shift, sample displacement, preferred orientation, finite crystallite size/strain effects, impurity peaks, peak overlap, and detector/q calibration artifacts.
  Experiment: add one perturbation family at a time and evaluate robustness on experimental or intentionally shifted validation patterns.

  Current implementation:
  - Store sparse `xrd_disc.q` and `xrd_disc.iq` peak lists in HDF5.
  - Generate continuous PXRD conditions at batch time to avoid storing many dense augmented traces.
  - Support Nyquist-style q-grid selection through `nyquist_points_per_fwhm`.
  - Support q shift, q scaling, peak intensity jitter, peak dropout, background, impurity peaks, particle-size broadening, peak asymmetry, noise, masking, and final normalization.

  Still needed:
  - hkl-aware preferred-orientation augmentation. This requires storing hkl metadata from XRD calculation; q/iq alone is not enough for a physically meaningful preferred-orientation transform.
  - Experimental-background templates or Chebyshev background coefficients if we want richer background distributions than the current smooth random baseline.
  - A calibration sweep to choose q-grid size from the minimum useful FWHM instead of blindly preserving the old dense `qstep=0.01`.

- P0 - Train with hard negatives and ambiguous PXRD neighborhoods.
  Many structures can have similar diffraction patterns. Construct batches or auxiliary tasks where the model must distinguish near-neighbor patterns, polymorphs, same-composition structures, and decoys with similar peak positions.
  Experiment: build nearest-neighbor sets in PXRD embedding space and add contrastive ranking loss.

- P1 - Expand synthetic data with controlled perturbations of structures.
  Generate physically plausible variants through small lattice/coordinate perturbations, symmetry lowering/restoration, supercell reductions, and composition-preserving distortions, then filter by validity and PXRD similarity.

- P1 - Domain adaptation from simulated to real PXRD.
  If real-world CSP is the goal, create a held-out real/experimental benchmark and add augmentation specifically targeted at the sim-to-real gap.

- P2 - Data quality scoring.
  Before increasing data size, score CIFs for parseability, charge/occupancy sanity, missing fields, extreme cell values, duplicate structures, and tokenization anomalies. Train ablations on clean-only vs full data.

### Tokenization and CIF representation

- P0 - Canonicalize CIF output more aggressively.
  Normalize field order, numeric precision, symmetry representation, atom ordering, and whitespace so the model spends less capacity on arbitrary formatting.

  Initial minicif DSL:
  `<mcif> Ni Co Fe cs_7 sg_225 cell a b c alpha beta gamma <atom> Ni mult x y z occ <atom> Co mult x y z occ </mcif>`

  Current implementation:
  - The prefix after `<mcif>` is the set of constituent elements, without stoichiometry, terminated by the first `cs_*` token.
  - `cs_1` through `cs_7` represents the crystal system.
  - `sg_1` through `sg_230` represents the space group number.
  - `cell` contains `a b c alpha beta gamma` with configurable decimal precision.
  - Each `<atom>` row contains `element multiplicity fract_x fract_y fract_z occupancy`.
  - Atom rows are sorted deterministically by atomic number, multiplicity, fractional coordinates, and occupancy.
  - The element-prefix order is configurable; later training can add permutation augmentation so the model does not overfit arbitrary element order.

  Not yet wired:
  - Minicif training/evaluation configs.
  - Minicif -> full CIF rendering for evaluation/generation.

  Current constrained decoding:
  - `sg_*` candidates are restricted to the emitted `cs_*` crystal-system range.
  - Atom-site element candidates after `<atom>` are restricted to the constituent elements emitted after `<mcif>`.
  - `</mcif>` is treated as the minicif generation stop token, while `<pad>` remains a stripped stop token.
  - The minicif small config enables constrained generation by default through `minicif_constrained_decoding`.

- P0 - Add grammar-aware decoding or constrained token masks.
  Many invalid generations can be prevented by masking impossible next tokens in known CIF contexts: field names, loop lengths, numeric formats, element symbols, and newline boundaries.
  Experiment: start with lightweight masks for element symbols and numeric fields, then expand.

  Current implementation:
  - Added a lightweight minicif state mask for canonical field order, numeric fields, crystal-system-to-space-group consistency, and atom-element membership in the prefix composition.
  - The mask is wired into `Decifer.generate*` for minicif checkpoints without changing training loss.

- P1 - Numeric tokenization for crystallographic values.
  Character-like numeric generation is inefficient and brittle. Consider digit-position tokens, quantized numeric bins, or structured numeric heads for cell parameters and coordinates.

- P1 - Separate structure semantics from serialization.
  Train on an intermediate JSON-like or table-like representation, then render CIF deterministically. This may sharply reduce syntax errors at the cost of building a renderer/parser path.

### Evaluation and validation

- P0 - Define a minicif evaluation suite.
  Minimum metrics: token loss, CIF parse rate, pymatgen structure construction rate, composition match, space-group/crystal-system match when available, cell-parameter error, PXRD agreement, duplicate rate, and generation wall time.

  Current implementation:
  - Added `bin/visualize_minicif.py`, which loads a minicif checkpoint and validation/test HDF5 splits, plots learning curves, generates candidates, computes parse rate, candidate and best-of-K match rate, Rwp, RMSD, composition match, space-group accuracy, and crystal-system accuracy.
  - Outputs per-candidate CSV, per-split summary CSV/JSON, `learning_curves.png`, `metric_summary.png`, and `rwp_distribution.png` when valid Rwp values are available.

- P0 - Evaluate with multiple samples per PXRD.
  Because the inverse problem is ambiguous, top-1 generation is too narrow. Report best-of-K and diversity-aware metrics for K values such as 4, 16, and 64.

- P1 - Use chemically meaningful splits.
  Random splits can overestimate performance if near-duplicates or same-composition structures leak across train/val/test. Add splits by composition system, prototype/family, reduced formula, and PXRD-nearest-neighbor distance.

- P1 - Track calibration of uncertainty.
  If the model samples multiple plausible structures, evaluate whether probability/ranking correlates with structure validity and PXRD agreement.

- P2 - Add regression tests around batch packing and conditioning alignment.
  The current debug assertion catches one failure mode, but a tiny deterministic unit test should pin down sequence truncation and condition-vector mapping.

### Generation and inference

- P0 - Rerank generated CIFs by forward-simulated PXRD agreement.
  Generate K candidates, parse each valid CIF, simulate PXRD, and rerank by agreement to the input pattern plus syntax/chemistry penalties.

- P0 - Add constrained decoding for CIF syntax and chemistry.
  Even simple masks for line starts, element symbols, numeric tokens, and loop consistency could reduce invalid samples before reranking.

- P1 - Use iterative repair.
  For invalid CIFs, run a small repair loop: parse error -> targeted regeneration of the broken section -> validate again. This may be cheaper than increasing K.

- P1 - Condition-strength and guidance experiments.
  If minicif separates CIF LM and PXRD conditioning, try classifier-free guidance-style dropout of conditioning during training, then guide generation by scaling conditional logits.

- P2 - Candidate diversity controls.
  Add systematic temperature/top-p/top-k sweeps and diversity penalties so best-of-K sampling explores genuinely different structures rather than formatting variants.

### Experiment infrastructure

- P0 - Create a tiny committed smoke dataset or synthetic test fixture.
  A minimal HDF5 fixture would let training, evaluation, checkpoint resume, and conditioning alignment be tested without access to NOMA data.

  Current implementation:
  - Added `bin/prepare_minicif_dataset.py`, a direct raw-CIF-to-HDF5 path for minicif experiments.
  - It avoids the legacy preprocessed/xrd/cif_tokens pickle directories and writes train/val/test HDF5 files directly.
  - It writes compact minicif tokens, sparse PXRD peaks, space group, crystal system, metadata, and optional failures.

- P0 - Add structured experiment logging.
  Write metrics to CSV/JSONL alongside checkpoints, including tokens seen, learning rate, gradient norm, throughput, validation mode, and augmentation settings.

- P1 - Add resume and checkpoint tests.
  Verify that optimizer state, iteration count, best model, patience counter, and metadata survive resume.

- P1 - Standardize ablation configs.
  Keep each config focused: baseline, no conditioning, clean-only PXRD, augmentation curriculum, cross-attention conditioning, grammar-constrained decoding.

- P2 - Track compute-normalized comparisons.
  Report performance versus tokens seen, wall time, and parameter count, not just final validation loss.
