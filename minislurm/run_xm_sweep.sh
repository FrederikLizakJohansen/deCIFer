#!/bin/bash

set -euo pipefail

SETTING="${1:-equal-updates}"
REPRESENTATION="${REPRESENTATION:-peak}"
SIZE="${SIZE:-medium}"
BASE_ITERS="${BASE_ITERS:-100000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-models/minicif_v2/xm/${REPRESENTATION}/standard/sweeps}"

if [[ "${SETTING}" != "equal-updates" && "${SETTING}" != "equal-compute" ]]; then
  echo "usage: $0 {equal-updates|equal-compute}" >&2
  exit 2
fi

for K in 1 2 4 8; do
  if [[ -n "${CONFIG:-}" ]]; then
    CONFIG_PATH="${CONFIG}"
  elif [[ "${K}" -eq 1 ]]; then
    CONFIG_PATH="configs/config_v2/${REPRESENTATION}/standard/${SIZE}.yaml"
  else
    CONFIG_PATH="configs/config_v2/xm/${REPRESENTATION}/standard/k${K}/${SIZE}.yaml"
  fi
  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "missing XM sweep config: ${CONFIG_PATH}" >&2
    exit 2
  fi
  ITERS="${BASE_ITERS}"
  if [[ "${SETTING}" == "equal-compute" && "${K}" -gt 1 ]]; then
    ITERS=$((BASE_ITERS * 3 / (K + 3)))
  fi
  sbatch minislurm/train_minicif_v2.sh \
    --config "${CONFIG_PATH}" \
    --xm-best-of-k "${K}" \
    --max-iters "${ITERS}" \
    --out-dir "${OUTPUT_ROOT}/${SETTING}/k${K}/${SIZE}"
done
