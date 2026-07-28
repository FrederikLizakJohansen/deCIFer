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
