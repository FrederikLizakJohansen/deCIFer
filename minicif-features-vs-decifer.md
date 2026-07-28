# minicif features compared to deCIFer

This file lists the core minicif changes relative to the original deCIFer workflow. It intentionally excludes visualization/reporting scripts and SLURM wrappers.

## Compact structure representation

- Added a compact minicif DSL instead of generating full CIF text.
- Represented each structure as:
  `<mcif> elements cs_* sg_* cell a b c alpha beta gamma <atom> element multiplicity x y z occupancy </mcif>`
- Removed full CIF headers, free-form CIF tags, symmetry-operation text, and formatting variability from the learning target.
- Added explicit start, atom-row, cell, and end tokens for the compact representation.
- Added configurable numeric precision for cell, coordinate, and occupancy values.
- Added a versioned `minicif_v2` representation without changing existing
  `minicif` checkpoints or datasets.
- `minicif_v2` includes an integral reduced formula and replaces multiplicity-only
  atom rows with explicit Wyckoff letters and one representative site per orbit.
- V2 canonicalization refines structures to a conventional symmetry setting;
  reconstruction expands sites with `Structure.from_spacegroup` and validates
  composition, space group, and declared Wyckoff letters.

## Tokenization and canonicalization

- Added `MinicifTokenizer`, with explicit tokens for elements, seven crystal systems, 230 space groups, digits, signs, decimal point, and spaces.
- Added deterministic CIF-to-minicif canonicalization.
- Canonicalization extracts:
  - constituent elements
  - crystal system
  - space-group number
  - lattice parameters
  - asymmetric-unit atom rows
- Atom rows are deterministically sorted by element, multiplicity, fractional coordinates, and occupancy.
- Constituent element order is configurable.

## Minicif dataset preparation

- Added direct raw-CIF to minicif HDF5 preparation.
- Added `.pkl.gz` raw CIF bundle support.
- Added resumable dataset preparation checkpoints.
- Added deterministic raw-subset creation for smaller training sets.
- Stored compact minicif tokens alongside sparse PXRD peak lists and metadata.

## PXRD storage and augmentation

- Store sparse peak positions/intensities instead of dense precomputed augmented traces.
- Added optional BraggCalculator sparse-pattern generation with automatic
  pymatgen fallback and persisted backend provenance.
- Reject overlength targets before diffraction generation; resumed legacy
  checkpoints are filtered when splits are written.
- Record-batched training uses a context-compatible dataset view and reports
  excluded overlength structures without rewriting the prepared data.
- Keep prepared PXRD data clean and sparse; artifacts are sampled during
  training instead of materialized into HDF5.
- Use BraggCalculator 0.3.0's device-native batch APIs for peak transforms and
  dense rendering, including a shared artifact realization for both hybrid
  encoder branches.
- Added Nyquist-style q-grid control through `nyquist_points_per_fwhm`.
- Added training-time calibration, peak-intensity, TCH profile, crystallite-size,
  microstrain, background, amorphous-hump, spurious-peak, Gaussian/correlated/
  Poisson-noise, detector-mask, saturation, and normalization artifacts.
- Added external artifact YAML files, an unseeded full training profile, and a
  fixed-seed evaluation profile.
- Preserve the flat legacy augmentation fields by mapping supported values to
  BraggCalculator pseudo-Voigt artifacts.
- Do not apply the old axial-asymmetry parameter in q space. Physically
  meaningful preferred orientation remains pending because the current dataset
  does not store unmerged HKL and lattice metadata.

## Conditioning architecture

- Kept the original single-vector MLP conditioning path available.
- Added configurable condition encoders with
  `condition_encoder: mlp|conv|patch|peak|peak_fourier|hybrid`.
- Added `condition_n_tokens`, allowing PXRD conditioning to use multiple non-generated condition tokens per minicif record.
- Added a 1D convolutional PXRD encoder over dense q-grid intensity traces.
- The conv encoder adaptively pools q-space to latent condition tokens and projects them to transformer width.
- Added an encoder-heavy hierarchical convolution that downsamples q before its
  widest stages, then applies bidirectional self-attention to the pooled PXRD
  tokens. Its reference config assigns 53.6% of all parameters to PXRD encoding.
- Added a sparse peak-list encoder over `xrd_disc.q` and `xrd_disc.iq`.
- Added `peak_fourier`, which encodes absolute q with fixed Fourier features and
  pools variable-length peak lists through learned latent queries without first
  materializing a dense PXRD trace.
- The Fourier peak encoder can apply bidirectional self-attention to its pooled
  tokens for encoder-heavy sparse models. The reference sparse-heavy config
  assigns about 55% of all parameters to PXRD encoding.
- Peak-list q positions are normalized against the configured training q range rather than the per-sample maximum q, preserving absolute q-position information.
- Added a hybrid PXRD encoder that concatenates dense-trace tokens and peak-list tokens.
- Encoder-heavy hybrid models use a hierarchical dense branch and latent
  self-attention in both PXRD branches.
- Organized the v2 training matrix under `configs/config_v2`, with peak, dense,
  and hybrid representations in standard/heavy small, medium, and large tiers.
- Condition tokens are inserted at each `<mcif>` start, preserving packed-batch condition alignment.
- Added optional true cross-attention from generated minicif tokens into PXRD memory tokens through `condition_cross_attention`.
- Added `condition_cross_attention_every_n_layers` to control how often transformer blocks attend to PXRD memory.
- Cached the fixed cross-attention keys and values during autoregressive
  generation, alongside the causal self-attention cache.
- Expanded evaluation reports with linear-scale learning curves, crystal-system
  metric summaries, best-Rwp CDFs, Rwp/RMSD diagnostics, and default
  PXRD-plus-structure examples stratified by reference crystal system.

## Attention and packing

- Preserved boundary masking for packed records.
- Extended conditioned attention masking to support multiple condition tokens per packed minicif record.
- Added a regression test that verifies a token in one packed minicif record cannot attend to tokens from a previous packed record.
- Added cross-attention masking so each packed minicif record attends only to its own PXRD memory tokens.
- Added handling for packed continuation blocks without an in-block `<mcif>` start token.
- Added record-aligned batching so every row contains one complete structure and
  its matching PXRD condition. This removes cross-record continuation blocks and
  allows the pure causal SDPA/Flash Attention path without a dense `T x T` group mask.
- Added length buckets and a padded-model-token budget per microbatch to reduce
  padding waste and make throughput comparisons explicit.

## Grammar-aware generation

- Added constrained minicif decoding.
- Restricts `sg_*` choices to the valid range for the already emitted `cs_*` crystal system.
- Restricts `<atom>` element choices to the constituent elements emitted in the minicif prefix.
- Uses minicif-specific stop behavior with `</mcif>` and `<pad>`.
- Added v2 constraints for formula counts, Wyckoff tokens, occupancy, cell-field
  order, crystal-system/space-group compatibility, and constituent atom elements.
- Lattice decoding deterministically fixes required angles and repeats equal cell
  lengths for conventional crystal-system settings.
- A constituents-only prefix remains valid: the model generates the formula when
  stoichiometry is not supplied at inference.

## Typed output heads

- Added optional element, symmetry, numeric, and control-token projections for
  `minicif_v2`. Each head scores only its static vocabulary partition and logits
  are reassembled for the same next-token cross-entropy objective.
- Numeric values remain tokenized rather than being regressed as continuous
  coordinates.

## Minicif-to-structure conversion

- Added parsing of generated minicif strings into structured fields.
- Added conversion from minicif to `pymatgen.Structure` using the emitted cell, space group, and atom rows.
- This enables structure-level validation and PXRD comparison without regenerating legacy CIF text first.

## Training workflow improvements

- Fixed gradient accumulation scaling.
- Made attention capture opt-in instead of always materializing attention maps.
- Improved deterministic seeding for Python, NumPy, PyTorch, samplers, and workers.
- Made validation use clean deterministic PXRD conditions rather than stochastic training augmentation.
- Added explicit conditioning alignment checks for packed batches.
- Improved checkpoint behavior so configured periodic saves work without relying only on validation.
- Added richer run metadata in checkpoints and `run_metadata.yaml`.
- Added structured training metrics output:
  - `metrics.jsonl`
  - `metrics.csv`
  - train/eval events
  - train and validation loss
  - learning rate
  - step time
  - token throughput
  - gradient norm
  - GPU memory
  - q-grid and condition-encoder settings
- Added a contrastive PXRD encoder pretraining workflow:
  - two independently augmented PXRD views per structure
  - shared condition encoder plus projection head
  - NT-Xent contrastive loss
  - checkpoint containing `encoder_state` for initializing minicif training
  - optional downstream freezing through `freeze_pretrained_condition_encoder`
  - live diagnostics through `contrastive_live.png`, `latest_metrics.json`, and `contrastive_metrics.csv`
- Added fused AdamW as an opt-in CUDA setting and separate useful-token versus
  padded-model-token throughput metrics.
- Added ready-to-run v2 small/medium training configs and an optional Fourier peak
  encoder pretraining config.
- Added a v2 dataset/config preflight that reports token-length percentiles and
  validates sampled token, formula, symmetry, Wyckoff expansion, and PXRD records.
- Training now fails before model initialization when v2 representation metadata,
  block size, or token budget is incompatible with the prepared splits.

## Evaluation and ablation workflow

- Added small conditioning ablation configs for:
  - no conditioning
  - dense MLP insertion
  - dense conv insertion
  - peak-list insertion
  - hybrid dense-plus-peak insertion
  - dense conv cross-attention
  - hybrid dense-plus-peak cross-attention
- Added a sequential ablation runner for local and SLURM runs.
- Extended minicif evaluation to support multiple prompt modes in one run:
  - `pxrd`
  - `pxrd-elements`
  - `pxrd-elements-cs`
  - `pxrd-elements-cs-sg`
- Added formula-aware v2 modes: `pxrd-stoichiometry`,
  `pxrd-stoichiometry-cs`, and `pxrd-stoichiometry-cs-sg`.
- Added formula accuracy to v2 evaluation summaries.
- Added generation metrics for finish rate, generated token count, extra elements, and missing elements.
- Added README instructions for running the ablations and evaluating checkpoints on the cluster.
