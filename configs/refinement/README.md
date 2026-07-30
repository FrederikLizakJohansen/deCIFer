# BraggCalculator refinement

Pass a refinement YAML to evaluation:

```bash
python bin/visualize_minicif.py \
  --checkpoint models/minicif_v2/peak/standard/medium/ckpt.pt \
  --dataset-dir data/noma_minicif_v2 \
  --refinement-config configs/refinement/quick.yaml
```

The top-level options are:

- `enabled`: run refinement for every valid generated candidate.
- `domain`: `q` or `two_theta`.
- `radiation`: `xray` or `neutron`.
- `wavelength`: wavelength in angstrom. `null` inherits `--wavelength`.
- `device`: Torch device. `null` inherits `--device`.
- `policy`: arguments used to build a BraggCalculator `RefinementPolicy`.
- `species_assignment`: arguments used to build an optional
  `SpeciesAssignmentConfig`.

`policy.preset` accepts `quick`, `cautious`, or `robust`. Other keys match
`RefinementPolicy`, including `refine_lattice`, `refine_coordinates`,
`occupancy_mode`, `refine_b_iso`, `refine_u_aniso`, restraints, profile
settings, and restart settings.

For exact parameter-group control, replace the preset stages:

```yaml
policy:
  preset: quick
  refine_coordinates: true
  stages:
    - name: scale/background
      active: [scale, background]
      steps: 40
      learning_rate: 0.04
    - name: structure
      active: [scale, background, zero_shift, profile, lattice, coordinates]
      steps: 100
      learning_rate: 0.01
```

Enable discrete species assignment before continuous refinement with:

```yaml
species_assignment:
  enabled: true
  search: auto
  fixed_sites: []
  max_candidates: 128
  continuous_top_k: 4
  composition_preserving: true
```

The remaining keys map directly to BraggCalculator 0.4.1
`SpeciesAssignmentConfig`.

CLI overrides are available for common experimental settings:

```text
--refine
--refinement-config PATH
--refinement-domain {q,two_theta}
--refinement-radiation {xray,neutron}
--refinement-wavelength ANGSTROM
--refinement-device DEVICE
--refinement-policy {quick,cautious,robust}
```

`--refine` enables the quick policy without requiring a YAML file.

## Included presets

| Config | Continuous optimization | Intended use |
| --- | --- | --- |
| `quick.yaml` | 200 steps, one start | Fast full-dataset baseline |
| `cautious.yaml` | 550 steps, one start | More stable lattice/profile refinement |
| `robust.yaml` | 420 coarse-to-fine steps, three restarts | Difficult profiles and final reporting |
| `cautious_coordinates.yaml` | 700 steps including restrained coordinates | Test whether positional refinement improves generated structures |
| `cautious_species_assignment.yaml` | Discrete site-species search followed by cautious refinement | Candidates with correct composition and uncertain site assignment |

The step counts describe the built-in BraggCalculator 0.4.1 policies. Runtime
also depends on structure size, profile length, device, and species-search
candidate count. Coordinate and species-assignment presets change more of the
candidate structure, so compare their refined structural metrics alongside Rwp.

Evaluate the same checkpoint into separate report directories when comparing
policies. The evaluation checkpoint signature prevents results from different
refinement configurations from being mixed:

```bash
for preset in quick cautious robust cautious_coordinates; do
  python bin/visualize_minicif.py \
    --checkpoint models/minicif_v2/peak/standard/medium/ckpt.pt \
    --dataset-dir data/noma_minicif_v2 \
    --out-dir "models/minicif_v2/peak/standard/medium/refinement_${preset}" \
    --splits test \
    --prompt-modes pxrd-elements \
    --num-reps 8 \
    --refinement-config "configs/refinement/${preset}.yaml"
done
```

Refinement adds scalar columns to `minicif_generation_metrics.csv` and writes
`minicif_refinement_results.jsonl.gz`. Each JSONL record contains candidate
identity, initial and refined fit statistics, convergence data, refined
parameters, warnings, objective and stage history, observed and calculated
profiles, residuals, starting and refined CIFs, diagnostics, provenance, and
species-assignment results. Candidate-specific failures are stored as error
records and evaluation continues.

Replay successful refinements in the example plotter:

```bash
python bin/plot_minicif_examples.py \
  --report-dir models/minicif_v2/peak/standard/medium/refinement_cautious \
  --show-refined
```

These figures contain reference, generated, and refined PXRD profiles and
structures. The script streams the selected records from
`minicif_refinement_results.jsonl.gz`.
