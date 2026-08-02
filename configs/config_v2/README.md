# minicif_v2 model configs

Model training configs are organized by PXRD representation, parameter
allocation, and model size:

```text
{peak,dense,hybrid}/{standard,heavy}/{small,medium,large}.yaml
```

`standard` assigns most parameters to the autoregressive minicif decoder.
`heavy` assigns most parameters to the PXRD encoder and uses a compact decoder.

All model outputs are written below:

```text
models/minicif_v2/{representation}/{allocation}/{size}
```

Approximate parameter ranges across the three representations:

| Size | Standard | Encoder-heavy |
| --- | ---: | ---: |
| small | 1.1-1.2M | 1.6-1.8M |
| medium | 13.1-13.7M | 7.7-8.0M |
| large | 30.3-31.5M | 21.1-23.6M |

Encoder-heavy configs assign 52-60% of their parameters to PXRD encoding.
Dense and hybrid models use `configs/xrd_artifacts/full_training.yaml`.
Peak models use sparse q-position and intensity perturbations.

## Explorative Modeling experiment

Forward XM can be enabled for any record-batched v2 config:

```yaml
xm_best_of_k: 2
```

`xm_best_of_k: 1` is the unchanged baseline. Use
`minislurm/run_xm_sweep.sh` for the controlled `K = 1, 2, 4, 8` comparison.
The implementation, compute budget, metrics, and interpretation are documented
in `XM-investigation.md`.

Self-contained XM configs use this hierarchy:

```text
xm/{peak,dense,hybrid}/standard/{k2,k4,k8}/{small,medium,large}.yaml
```

The `peak` representation uses `condition_encoder: peak_fourier`. Outputs are
written to:

```text
models/minicif_v2/xm/{representation}/standard/{k2,k4,k8}/{size}
```

For example:

```bash
sbatch minislurm/train_minicif_v2.sh \
  --config configs/config_v2/xm/hybrid/standard/k2/medium.yaml
```
