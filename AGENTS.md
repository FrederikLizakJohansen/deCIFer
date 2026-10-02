# Repository Guidelines

## Project Structure & Module Organization

This repository contains the deCIFer Python package for CIF generation from powder diffraction data. Core library code lives in `decifer/`, including the model, tokenizer, dataset, and utility functions. Command-line workflows are in `bin/` (`train.py`, `evaluate.py`, `prepare_dataset.py`, and experiment helpers). Training and evaluation YAML files are in `configs/`; cluster launch scripts are in `slurm/`; exploratory analysis is in `notebooks/`. Paper-specific configs, figure scripts, and result artifacts are grouped under `follow-up-paper-Tackling-Real-World-CSP/`. Keep large data, checkpoints, and generated outputs outside git-tracked source paths, typically under `data/` and `models/`.

## Build, Test, and Development Commands

Use Python 3.9, which is the version documented for this project.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install torch numpy pandas matplotlib seaborn pyYAML tqdm omegaconf h5py pymatgen periodictable scikit-learn
```

Prepare serialized data from raw NOMA input:

```bash
python bin/prepare_dataset.py --data-dir data/ --name noma --all --raw-from-gzip
```

Run a small training configuration for smoke testing:

```bash
python bin/train.py --config configs/deCIFer_NOMA_small_config.yaml
```

Use `python bin/evaluate.py --help` and `python bin/run_protocol.py --help` before evaluation workflows, because inputs depend on local checkpoints and datasets.

## Coding Style & Naming Conventions

Follow the existing Python style: 4-space indentation, standard-library imports before third-party imports, and snake_case for functions, variables, and module names. Keep dataclass-based configuration fields explicit and aligned with YAML config keys. Prefer adding reusable logic to `decifer/` and keeping `bin/` scripts as workflow entry points. Name new configs descriptively, for example `deCIFer_NOMA_small_config.yaml` or `ablation_cubic_decifer.yaml`.

## Testing Guidelines

There is currently no dedicated test suite in the repository. Validate changes with targeted smoke runs using the smallest available config and, when relevant, a tiny `--debug-max` dataset preparation run. For model, tokenizer, or dataset changes, add lightweight tests under a new `tests/` directory if possible, and document any data or checkpoint assumptions.

## Commit & Pull Request Guidelines

Recent commits use short, imperative, lowercase summaries such as `fix dataset path to data/noma` and `add small deCIFer config for testing`. Keep commits focused on one behavior or workflow update. Pull requests should describe the affected scripts/configs, list commands run, note required local data or checkpoints, and include screenshots or exported figures only when notebook or figure output changes.

## Agent-Specific Instructions

Do not overwrite large datasets, checkpoints, notebook outputs, or paper artifacts unless explicitly requested. Before changing configs, check path assumptions against `README.md` and the relevant workflow script.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
