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

Refinement adds scalar columns to `minicif_generation_metrics.csv` and writes
`minicif_refinement_results.jsonl.gz`. Each JSONL record contains candidate
identity, initial and refined fit statistics, convergence data, refined
parameters, warnings, objective and stage history, observed and calculated
profiles, residuals, starting and refined CIFs, diagnostics, provenance, and
species-assignment results. Candidate-specific failures are stored as error
records and evaluation continues.
