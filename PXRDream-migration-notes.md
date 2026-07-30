# PXRDream Migration Notes

## Project name

**PXRDream**

Working title:

**PXRDream: From Noisy Powder Diffraction to Refined Crystal Structures**

PXRDream will cover the complete path from artifacted experimental PXRD to a
refined structure:

1. Detect peak positions and intensities in noisy experimental patterns.
2. Infer candidate crystal structures from the detected peaks and available
   chemical information.
3. Refine generated candidates against the observed diffraction pattern.

## Current decision

Development will continue on the `minicif` branch of the existing deCIFer
repository until the model architecture, representation, training workflow,
evaluation protocol, and public interfaces are stable.

No repository migration or broad code rename is planned during this stage.
`minicif` remains the internal working name where changing it would create
unnecessary churn.

## Possible repository migration

When the system is stable, create an empty `PXRDream` repository and push the
current `minicif` branch as its `main` branch:

```bash
git push git@github.com:FrederikLizakJohansen/PXRDream.git \
  refs/heads/minicif:refs/heads/main
```

This leaves the existing deCIFer repository unchanged and gives PXRDream a
separate history-preserving repository. The original deCIFer `main` branch does
not need to be copied because it remains available in the original repository
and its shared history is already reachable from the PXRDream branch.
