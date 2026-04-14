# Code Review Notes

Last updated: 2026-04-05

## Scope

These notes summarize a read-through of the current repository structure and the main pitfalls found in the preprocessing, training, model, and evaluation pipeline.

## High-level picture

- `bin/prepare_dataset.py` builds the serialized dataset from raw CIFs.
- `bin/train.py` trains a GPT-style CIF generator.
- `decifer/decifer_model.py` contains the Transformer model and generation code.
- `bin/evaluate.py` runs generation plus structural/XRD evaluation.
- `bin/collect_evaluations.py` aggregates evaluation artifacts into analysis-ready data.

## Findings And Status

### 1. `num_reps` in evaluation

- File: [bin/evaluate.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/evaluate.py)
- Current issue:
  The current implementation does not actually support `num_reps > 1` correctly. It generates one sample, then indexes the result as if multiple samples had been produced.
- Notes:
  Variable-length autoregressive generation makes true batched multi-rep generation awkward here.
- Preferred direction:
  Do not optimize for batched variable-length generation right now. Either:
  generate repetitions serially with batch size 1, or
  explicitly constrain/document evaluation to batch size 1 for repeated generations.
- Status:
  Fixed on 2026-04-05 by generating repetitions serially with batch size 1.

### 2. Evaluation bookkeeping / resume logic

- File: [bin/evaluate.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/evaluate.py)
- Current issue:
  Existing-file counting, dataset iteration bounds, per-sample skipping, and completion tracking are inconsistent.
- Consequences:
  Partial output folders can cause skipped samples, incorrect progress accounting, or hangs while waiting for more completed tasks than were actually submitted.
- Status:
  Fixed on 2026-04-05.

### 3. Gradient accumulation scaling

- File: [bin/train.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/train.py)
- Current issue:
  The loss is backpropagated once per micro-step without dividing by `gradient_accumulation_steps`.
- Consequences:
  The effective update size changes with the accumulation setting, which makes training behavior depend on accumulation in a non-standard way.
- Status:
  Fixed on 2026-04-05.

### 4. Train/val/test split sizing

- File: [bin/prepare_dataset.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/prepare_dataset.py)
- Current issue:
  The second `train_test_split()` uses `test_size=test_size` again instead of `val_size`, so the actual validation fraction does not match the printed numbers.
- Consequences:
  The dataset split is not what the script reports.
- Status:
  Fixed on 2026-04-05.

### 5. Flash-attention dropout during eval/generation

- File: [decifer/decifer_model.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/decifer_model.py)
- Current issue:
  `scaled_dot_product_attention()` is called with `dropout_p=self.dropout` directly.
- Consequences:
  In PyTorch, this can keep dropout active during eval/generation unless it is explicitly zeroed when `self.training == False`.
- Status:
  Fixed on 2026-04-05.

### 6. Collection aborts on first bad file

- File: [bin/collect_evaluations.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/collect_evaluations.py)
- Current issue:
  The exception handler immediately re-raises, so the logging/skip path below it is dead code.
- Consequences:
  One malformed evaluation file can stop the entire collection job.
- Status:
  Fixed on 2026-04-05.

### 7. `generate_and_print()` uses an undefined tokenizer

- File: [decifer/decifer_model.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/decifer_model.py)
- Current issue:
  `generate_and_print()` references `tokenizer`, but no local variable with that name exists in the method.
- Consequences:
  The helper will fail if it is called.
- Status:
  Fixed on 2026-04-05.

### 8. `TrainConfig` defines `block_size` twice

- File: [bin/train.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/train.py)
- Current issue:
  `block_size` is declared twice with different defaults.
- Consequences:
  The later value silently wins, which makes the earlier definition misleading.
- Status:
  Fixed on 2026-04-05.

## Suggested Next Pass

- Add a small regression test layer for training/evaluation bookkeeping, because most of these bugs were pipeline-level and would not be caught by syntax checks.
- Decide whether `debug_max` in evaluation should remain sample-based; it now behaves that way, which matches the CLI/help text better than the previous generation-count behavior.

## Workflow Review

### Current Flow

The repository currently has a workable end-to-end path, but the workflow is spread across multiple script-specific entry points:

- Data preparation: [bin/prepare_dataset.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/prepare_dataset.py)
- Training: [bin/train.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/train.py)
- Evaluation: [bin/evaluate.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/evaluate.py) and [bin/collect_evaluations.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/collect_evaluations.py)
- Synthetic artefact sweeps: [bin/ablation.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/ablation.py)
- Experimental PXRD protocols: [bin/experimental_pipeline.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/experimental_pipeline.py) and [bin/run_protocol.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/run_protocol.py)

This split is functional, but it makes experimentation harder because each script owns a lot of logic that should really be shared.

### Main Friction Points

#### 1. The same concepts are implemented multiple times

The following ideas recur in several places:

- model loading
- prompt construction
- PXRD conditioning setup
- generation loops
- CIF post-processing / symmetry repair
- evaluation metrics

This means changes to generation or model behavior need to be propagated manually across several scripts.

#### 2. Configuration is fragmented

The repo currently mixes several styles:

- `bin/train.py` uses `OmegaConf`
- `bin/ablation.py` uses direct YAML parsing
- `bin/evaluate.py` is mostly CLI flags
- `bin/experimental_pipeline.py` is mostly class arguments / ad hoc state

This makes it harder to compare runs, reuse settings, or create new experiment types cleanly.

#### 3. Path semantics are inconsistent

Different scripts expect different forms of dataset/model paths:

- training tends to expect a dataset root
- evaluation expects a serialized HDF5 file
- ablation has its own expectations

That creates avoidable friction and more script-local glue code.

#### 4. Scripts import each other

There are several cases of `bin/*` importing from other `bin/*` scripts. That is a sign that code which should live in the library layer currently lives in CLI entry points.

Example:

- `bin/ablation.py` imports from `bin.evaluate` and `bin.train`

The better structure is that `bin/*` should be thin wrappers over reusable `decifer/*` modules.

#### 5. Outputs are script-local instead of run-local

There is no single standard experiment run format that consistently stores:

- config
- metrics
- logs
- predictions
- provenance

Each script writes artifacts in its own way. That makes runs harder to compare and harder to resume or inspect later.

#### 6. The repo is optimized for individual scripts rather than composable workflows

Right now the mental model is:

- run this script for training
- another script for evaluation
- another script for robustness studies
- another script for experimental PXRD

This works for paper-specific execution, but it makes later expansion more expensive than it needs to be.

## Simplification Direction

The most useful simplification is to reorganize around one reusable experiment layer, while keeping the CLI entry points thin.

### Recommended Architecture

#### 1. Create a real shared workflow/library layer

Suggested modules:

- `decifer/config.py`
- `decifer/io.py`
- `decifer/generation.py`
- `decifer/evaluation.py`
- `decifer/workflows/prepare.py`
- `decifer/workflows/train.py`
- `decifer/workflows/evaluate.py`
- `decifer/workflows/ablate.py`
- `decifer/workflows/experimental.py`

The goal is that `bin/*` becomes a thin CLI layer that only:

- parses arguments
- loads config
- calls a workflow function

#### 2. Standardize configuration everywhere

Use one config style across workflows, preferably `OmegaConf` since it is already used in training.

Each workflow should support:

- a YAML config file
- CLI overrides
- one typed schema per workflow

This would make it much easier to compare training, evaluation, ablation, and experimental runs.

#### 3. Introduce a single run directory format

Each experiment run should write a standard structure such as:

- `run.yaml`
- `metrics.json`
- `metadata.json`
- `logs/`
- `artifacts/`
- `predictions/`

This would provide a uniform way to inspect results regardless of workflow type.

#### 4. Pull generation into one reusable engine

A shared generation layer should own:

- checkpoint loading
- prompt assembly
- conditioning vector preparation
- serial `n_reps` generation
- CIF decode
- CIF repair / symmetry restoration

This should be reused by:

- `bin/evaluate.py`
- `bin/ablation.py`
- `bin/experimental_pipeline.py`

#### 5. Pull evaluation into one reusable engine

A shared evaluator should own:

- validity checks
- structural comparison / RMSD
- CIF statistics
- XRD metrics

This reduces duplication and makes future metric additions much safer.

#### 6. Normalize dataset access behind one adapter

A shared dataset abstraction should hide whether the input is:

- a dataset root
- a serialized `.h5`
- a filtered view/subset

This would remove a lot of script-local assumptions and path juggling.

#### 7. Stop using `bin/*` as dependency roots

Nothing in `bin/` should be the canonical home for reusable functions.

Instead:

- reusable logic lives in `decifer/*`
- `bin/*` only imports from `decifer/*`

## Design Goal

The desired workflow should be:

- define config
- run one command
- get one run folder
- reuse the same generator and evaluator stack whether the task is standard evaluation, ablation, or experimental PXRD

This is also the cleanest route for future model expansion, because new model variants would only need to plug into:

- model config
- generation adapter
- optional conditioning adapter

instead of requiring edits across several scripts.

## Refactor Roadmap

This should be done incrementally. The highest-leverage path is to centralize shared behavior first, then standardize configuration and outputs.

### Phase 1. Extract shared generation logic

Goal:

- remove duplicate model-loading and generation logic from script entry points

Tasks:

- create `decifer/generation.py`
- move model checkpoint loading there
- move prompt construction there
- move serial generation there
- move CIF decode / symmetry-fix logic there

Target consumers:

- `bin/evaluate.py`
- `bin/ablation.py`
- `bin/experimental_pipeline.py`

Expected benefit:

- one place to change generation behavior
- easier experimentation with prompt variants and model variants

### Phase 2. Extract shared evaluation logic

Goal:

- remove duplicated validity / RMSD / XRD scoring logic

Tasks:

- create `decifer/evaluation.py`
- move CIF statistics collection there
- move validity checks there
- move structure matching / RMSD there
- move PXRD metric computation there

Target consumers:

- `bin/evaluate.py`
- `bin/collect_evaluations.py`
- `bin/ablation.py`

Expected benefit:

- consistent metrics across workflows
- easier extension when new metrics are added

### Phase 3. Standardize config handling

Goal:

- make all workflows configurable in the same way

Tasks:

- create typed config schemas under `decifer/config.py` or `decifer/workflows/*`
- migrate `ablation.py` away from ad hoc YAML parsing
- allow `evaluate.py` to accept a config file in addition to CLI overrides
- keep CLI overrides for convenience

Expected benefit:

- easier reproducibility
- easier parameter sweeps
- less workflow-specific argument handling

### Phase 4. Standardize run outputs

Goal:

- make experiment artifacts inspectable and resumable in a uniform way

Tasks:

- define one run directory layout
- make workflows emit `run.yaml`, metrics, metadata, and artifacts consistently
- stop writing loosely related outputs in arbitrary script-specific locations

Expected benefit:

- simpler analysis
- easier reruns and comparisons
- better provenance

### Phase 5. Normalize dataset access

Goal:

- remove path-shape assumptions from workflow scripts

Tasks:

- create a shared dataset spec / adapter layer
- support dataset root, serialized HDF5, and filtered subsets through one interface
- keep filtering utilities separate from workflow code

Expected benefit:

- less path glue
- less duplicated dataset filtering logic
- easier future expansion to new dataset formats

### Phase 6. Thin out `bin/*`

Goal:

- make `bin/*` true entry points rather than implementation files

Tasks:

- move reusable logic out of `bin/*`
- keep `bin/*` focused on parsing args and invoking workflows
- eliminate script-to-script imports inside `bin/*`

Expected benefit:

- clearer project structure
- safer future maintenance
- easier testing of the real logic

## Suggested Execution Order

Recommended order:

1. Phase 1: shared generation
2. Phase 2: shared evaluation
3. Phase 6: thin out `bin/*` as the shared pieces become available
4. Phase 3: unify config handling
5. Phase 4: standardize run outputs
6. Phase 5: normalize dataset access

Reasoning:

- phases 1 and 2 remove the most duplication and unlock almost every later cleanup
- once that shared layer exists, the rest of the refactor becomes much lower risk

## Refactor Principle

The key principle for the refactor should be:

- keep current workflows working
- move logic into reusable library modules
- standardize behavior without rewriting everything at once

This should be treated as a staged extraction, not as a greenfield rewrite.

## Real Workflow Test Grocery List

These are the external artifacts needed later for real workflow smoke tests that exercise the actual scripts against realistic inputs.

### 1. Tiny serialized dataset bundle

Preferred contents:

- `serialized/train.h5`
- `serialized/val.h5`
- `serialized/test.h5`

Even 5 to 20 samples is enough.

Desired fields:

- `cif_name`
- `cif_string`
- `spacegroup`
- `cif_tokens`
- `xrd.q`
- `xrd.iq`

Purpose:

- smoke-test training data loading
- smoke-test evaluation against real serialized inputs
- validate prompt extraction and conditioning paths

### 2. One compatible model checkpoint

Requirements:

- does not need to be good
- only needs to load correctly with the current model code

Purpose:

- smoke-test checkpoint loading
- smoke-test generation path
- smoke-test evaluation output writing

### 3. One tiny experimental PXRD ZIP

Preferred contents:

- 1 target `.xy` or `.xye` file
- 1 background `.xy` or `.xye` file

Purpose:

- smoke-test `bin/experimental_pipeline.py`
- smoke-test `bin/run_protocol.py`

### 4. Known-good smoke commands

Need 2 to 4 commands that should still work after refactors.

Examples:

- one training smoke command
- one evaluation smoke command
- one ablation smoke command
- one experimental protocol smoke command

For each command, it is helpful to define what “success” means, e.g.:

- should produce `N` files
- should complete without crashing
- should write a pickle with expected top-level keys
- should create a run folder with expected artifacts

Exact metric values are not required unless golden-result tests are desired.

### 5. Optional raw-preprocessing fixture

Optional but helpful:

- a tiny raw CIF dataset or one raw `.pkl.gz` bundle

Purpose:

- smoke-test `bin/prepare_dataset.py`

### Minimum viable external bundle

The smallest practical set is:

1. a tiny serialized dataset
2. one checkpoint
3. one tiny experimental ZIP
4. a short list of known-good commands and expected outputs

## Current Local Test Layer

As of 2026-04-05, the repository now has a dependency-light `pytest` suite under `tests/`.

Current coverage:

- tokenizer behavior
- dataset key mapping / type conversion through a stubbed HDF5 layer
- evaluation bookkeeping and serial repetition handling through a stubbed evaluation environment
- flash-attention dropout behavior in train vs eval mode
- `generate_and_print()` tokenizer usage
- collection error handling for malformed evaluation artifacts

Current status:

- `pytest -q tests` passes in the local environment
- the current local test run is warning-free

Limitations:

- this suite does not replace real workflow smoke tests
- it does not yet validate end-to-end training, evaluation, ablation, or experimental PXRD runs against real artifacts

## Refactor Progress

### Phase 1 started on 2026-04-05

Implemented so far:

- created [decifer/generation.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/generation.py)
- moved shared prompt extraction there
- moved checkpoint loading there
- moved serial single-generation helper there
- moved CIF decode / symmetry-fix helpers there
- updated [bin/evaluate.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/evaluate.py) to use the shared generation module
- updated [bin/ablation.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/ablation.py) to use the shared generation module for prompt extraction, checkpoint loading, and CIF repair
- updated [bin/experimental_pipeline.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/experimental_pipeline.py) to use the shared generation module for checkpoint loading and CIF repair

Additional cleanup completed alongside this:

- removed the accidental `h5py`-based `sys` import from [decifer/decifer_model.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/decifer_model.py)
- removed the prompt-extraction warning path by centralizing prompt parsing in the shared generation module

Current interpretation:

- this is the first extraction step, not a complete workflow rewrite
- `bin/*` still contains substantial logic, but the dependency direction is now cleaner than before

### Phase 2 started and completed on 2026-04-05

Implemented so far:

- created [decifer/evaluation.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/evaluation.py)
- moved shared CIF statistics / validity extraction there
- moved shared Rwp and Wasserstein-based evaluation summarization there
- updated [bin/evaluate.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/evaluate.py) to use the shared evaluation module
- updated [bin/collect_evaluations.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/collect_evaluations.py) to use the shared evaluation summarizer
- updated [bin/ablation.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/ablation.py) to use shared validity helpers

Testing status after Phase 2:

- `pytest -q tests` passes
- `python -m py_compile decifer/evaluation.py bin/evaluate.py bin/collect_evaluations.py bin/ablation.py` passes

Current interpretation:

- the evaluation layer is now materially less duplicated
- `bin/evaluate.py` still contains orchestration logic, multiprocessing, and file I/O, but much less reusable evaluation behavior is trapped there than before

### Phase 3 started and completed on 2026-04-05

Implemented so far:

- created [decifer/config.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/config.py)
- added typed shared config dataclasses for evaluation, ablation, training, and protocol workflows
- added shared YAML loading and CLI override merging there
- updated [bin/evaluate.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/evaluate.py) to support `--config` plus CLI overrides through the shared config path
- updated [bin/ablation.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/ablation.py) to use the shared config path instead of ad hoc YAML loading
- updated [bin/train.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/train.py) to use the shared config path and removed the `OmegaConf` dependency
- updated [bin/run_protocol.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/run_protocol.py) to use the shared config path

Testing status after this step:

- `pytest -q tests` passes
- `python -m py_compile decifer/config.py bin/evaluate.py bin/ablation.py bin/train.py bin/run_protocol.py` passes

Current interpretation:

- the major workflow entry points now resolve configuration through the same shared mechanism
- this removes one of the main sources of workflow fragmentation
- `bin/*` still contains a lot of orchestration code, so the next cleanup should focus on moving workflow logic itself into reusable modules rather than only standardizing config

### Workflow extraction phase completed on 2026-04-05

Implemented so far:

- created [decifer/workflows/](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/workflows)
- moved the current orchestration implementations for:
  - [evaluate](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/workflows/evaluate.py)
  - [ablation](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/workflows/ablation.py)
  - [train](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/workflows/train.py)
  - [run_protocol](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/workflows/run_protocol.py)
  - [experimental_pipeline](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/workflows/experimental_pipeline.py)
- reduced the corresponding `bin/*` entry points to thin wrappers that re-export workflow functions and invoke workflow `main` behavior
- reduced [bin/experimental_pipeline.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/experimental_pipeline.py) to a compatibility wrapper around the workflow module
- added a wrapper-level regression test so the old import surface remains covered

Testing status after this step:

- `pytest -q tests` passes with 15 tests
- `python -m py_compile decifer/workflows/__init__.py decifer/workflows/evaluate.py decifer/workflows/ablation.py decifer/workflows/train.py decifer/workflows/run_protocol.py decifer/workflows/experimental_pipeline.py bin/evaluate.py bin/ablation.py bin/train.py bin/run_protocol.py bin/experimental_pipeline.py` passes

Current interpretation:

- the dependency direction is now substantially cleaner
- the CLI entry points are much thinner than before
- reusable workflow orchestration no longer depends on implementation code living under `bin/*`
- the next architectural phase should focus on standardizing run-output layout rather than continuing the `bin/*` extraction

### Phase 4 started and completed on 2026-04-05

Implemented so far:

- created [decifer/io.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/io.py) as a shared run-layout/output utility
- standardized every workflow run around the same root structure:
  - `run.yaml`
  - `metadata.json`
  - `metrics.json`
  - `artifacts/`
  - `logs/`
  - `predictions/`
- updated [decifer/workflows/evaluate.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/workflows/evaluate.py) to resolve a run directory, emit run metadata, and write evaluation files under the standardized `predictions/` tree
- updated [decifer/workflows/ablation.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/workflows/ablation.py) to create a run directory from the configured output name and write the result pickle under `artifacts/`
- updated [decifer/workflows/run_protocol.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/workflows/run_protocol.py) to treat the protocol folder as a standardized run root and write generated pickles under `predictions/`
- updated [decifer/workflows/train.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/workflows/train.py) to treat `out_dir` as a standardized run root and to snapshot training metrics alongside checkpoints
- added [tests/test_io.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/tests/test_io.py) to lock the new output contract in place

Testing status after this step:

- `pytest -q tests` passes with 17 tests
- `python -m py_compile decifer/io.py decifer/workflows/evaluate.py decifer/workflows/ablation.py decifer/workflows/run_protocol.py decifer/workflows/train.py` passes

Current interpretation:

- run outputs are now much easier to inspect and compare across workflows
- training, evaluation, ablation, and protocol runs now share the same top-level artifact structure
- the next architectural phase should focus on dataset access normalization rather than output cleanup

### Phase 5 started and completed on 2026-04-05

Implemented so far:

- created [decifer/datasets.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/datasets.py) as a shared dataset resolution/adapter layer
- standardized dataset-path handling around one rule:
  - if the path ends in `.h5`, treat it as an explicit serialized dataset file
  - if the path is a directory, resolve the requested split from either `serialized/<split>.h5` or `<split>.h5`
- updated [decifer/workflows/evaluate.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/workflows/evaluate.py) to load datasets through the shared adapter and to support `dataset_split` when the configured dataset path is a root
- updated [decifer/workflows/ablation.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/workflows/ablation.py) to use the same adapter and `dataset_split` behavior
- updated [decifer/workflows/train.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/workflows/train.py) to resolve train/val/test via the shared split resolver instead of hardcoding path joins inline
- extended [decifer/config.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/config.py) so evaluation and ablation configs carry an explicit `dataset_split` field
- added [tests/test_datasets.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/tests/test_datasets.py) to lock the new root-vs-file resolution behavior in place

Testing status after this step:

- `pytest -q tests` passes with 22 tests
- `python -m py_compile decifer/datasets.py decifer/workflows/evaluate.py decifer/workflows/ablation.py decifer/workflows/train.py decifer/config.py` passes

Current interpretation:

- training, evaluation, and ablation now share one dataset access contract instead of each workflow owning its own path assumptions
- future dataset-format work can now happen behind the adapter layer rather than inside the workflows
- the next architectural step should focus on trimming remaining workflow-specific orchestration or unifying more of the experiment control surface

### Phase 6 started and completed on 2026-04-05

Implemented so far:

- created [decifer/training.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/training.py) as the new library home for the implementation-heavy training runtime
- created [decifer/experimental.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/experimental.py) as the new library home for the experimental PXRD pipeline class
- reduced [decifer/workflows/train.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/workflows/train.py) to a thin compatibility layer that re-exports the training surface and dispatches execution into the library module
- reduced [decifer/workflows/experimental_pipeline.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/workflows/experimental_pipeline.py) to a thin compatibility re-export
- updated [decifer/workflows/run_protocol.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/workflows/run_protocol.py), [bin/train.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/train.py), and [bin/experimental_pipeline.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/bin/experimental_pipeline.py) to point at the new library-level modules instead of implementation code under `workflows/`
- updated [tests/test_entrypoint_configs.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/tests/test_entrypoint_configs.py) so the entrypoint compatibility surface is still covered after the import move

Testing status after this step:

- `pytest -q tests` passes with 22 tests
- `python -m py_compile decifer/training.py decifer/experimental.py decifer/workflows/train.py decifer/workflows/experimental_pipeline.py decifer/workflows/run_protocol.py bin/train.py bin/experimental_pipeline.py` passes

Current interpretation:

- the workflow package is now much closer to what it should be: an entrypoint/orchestration layer rather than the primary home of large implementation bodies
- the heavy reusable logic now lives under top-level library modules in `decifer/`
- the next architectural step should focus on unifying more of the experiment control surface or reducing duplication inside the large library modules themselves

### Phase 7 started and completed on 2026-04-05

Implemented so far:

- refactored [decifer/training.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/training.py) from a script-style module into an explicit callable surface with:
  - `run_training(config)`
  - `main(argv=None)`
  - smaller helpers for augmentation config, training metrics, model args, and model/checkpoint initialization
- updated [decifer/workflows/train.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/workflows/train.py) to call the explicit training API instead of dispatching through `runpy`
- simplified [decifer/experimental.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/experimental.py) by extracting pure helper functions for:
  - active-element list construction
  - reading experimental ZIP data
  - signal standardization
  - experimental preprocessing
  - space-group-symbol lookup
- kept the public compatibility surface intact for the existing workflow and `bin/*` entry points

Testing status after this step:

- `pytest -q tests` passes with 22 tests
- `python -m py_compile decifer/training.py decifer/experimental.py decifer/workflows/train.py decifer/workflows/experimental_pipeline.py decifer/workflows/run_protocol.py bin/train.py bin/experimental_pipeline.py` passes

Current interpretation:

- training now has a real library-level execution API rather than only a script body
- the experimental pipeline is still large, but less of it is trapped inside one monolithic class
- the next simplification step should likely focus on either:
  - splitting `decifer/experimental.py` further into plotting vs preprocessing vs generation helpers, or
  - introducing a more uniform experiment-runner abstraction shared across train/evaluate/ablation/protocol

### Phase 8 started and completed on 2026-04-05

Implemented so far:

- added [configs/local_cpu_training_minimal.yaml](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/configs/local_cpu_training_minimal.yaml) as a minimal CPU-safe starter training config
- added [configs/local_cpu_evaluate_minimal.yaml](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/configs/local_cpu_evaluate_minimal.yaml) as a matching validation-preview evaluation config
- improved [decifer/training.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/decifer/training.py) to write live training metrics into `metrics.json` during the log loop, not only at checkpoint time
- added [apps/local_monitor.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/apps/local_monitor.py) as a dependency-light localhost monitor that can:
  - start and stop training
  - start and stop validation-preview evaluation against the latest checkpoint
  - display live training metrics from `metrics.json`
  - display evaluation progress and one-to-one validation comparison cards from generated evaluation files
- added [tests/test_local_monitor.py](/home/frederik/phd/papers/deCIFer-tmlr/deCIFer/tests/test_local_monitor.py) to cover the new monitor helpers

Testing status after this step:

- `pytest -q tests` passes with 24 tests
- `python -m py_compile apps/local_monitor.py decifer/training.py` passes

Current interpretation:

- the repo now has a workable local loop for CPU smoke training plus validation-preview inspection
- the monitoring surface is still intentionally lightweight, but it is now practical for local experimentation without notebooks or manual directory inspection
