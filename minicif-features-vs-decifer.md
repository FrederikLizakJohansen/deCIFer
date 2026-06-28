# minicif features compared to deCIFer

This file lists the core minicif changes relative to the original deCIFer workflow. It intentionally excludes visualization/reporting scripts and SLURM wrappers.

## Compact structure representation

- Added a compact minicif DSL instead of generating full CIF text.
- Represented each structure as:
  `<mcif> elements cs_* sg_* cell a b c alpha beta gamma <atom> element multiplicity x y z occupancy </mcif>`
- Removed full CIF headers, free-form CIF tags, symmetry-operation text, and formatting variability from the learning target.
- Added explicit start, atom-row, cell, and end tokens for the compact representation.
- Added configurable numeric precision for cell, coordinate, and occupancy values.

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
- Reconstruct continuous PXRD conditions during training.
- Added Nyquist-style q-grid control through `nyquist_points_per_fwhm`.
- Added training-time PXRD perturbations:
  - q shift
  - q scaling
  - peak intensity jitter
  - peak dropout
  - smooth background
  - impurity peaks
  - particle-size broadening
  - peak asymmetry
  - noise
  - masking
  - final normalization

## Conditioning architecture

- Kept the original single-vector MLP conditioning path available.
- Added configurable condition encoders with `condition_encoder: mlp|conv`.
- Added `condition_n_tokens`, allowing PXRD conditioning to use multiple non-generated condition tokens per minicif record.
- Added a 1D convolutional PXRD encoder over dense q-grid intensity traces.
- The conv encoder adaptively pools q-space to latent condition tokens and projects them to transformer width.
- Condition tokens are inserted at each `<mcif>` start, preserving packed-batch condition alignment.

## Attention and packing

- Preserved boundary masking for packed records.
- Extended conditioned attention masking to support multiple condition tokens per packed minicif record.
- Added a regression test that verifies a token in one packed minicif record cannot attend to tokens from a previous packed record.

## Grammar-aware generation

- Added constrained minicif decoding.
- Restricts `sg_*` choices to the valid range for the already emitted `cs_*` crystal system.
- Restricts `<atom>` element choices to the constituent elements emitted in the minicif prefix.
- Uses minicif-specific stop behavior with `</mcif>` and `<pad>`.

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
