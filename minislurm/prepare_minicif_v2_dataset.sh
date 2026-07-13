#!/bin/bash
#SBATCH --time 1-00:00:00
#SBATCH --job-name=prepare_mcif_v2
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/minicif_v2_prepare_%j.out

RAW_DIR="${RAW_DIR:-data/noma}"
OUT_DIR="${OUT_DIR:-data/noma_minicif_v2}"

mkdir -p logs
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python bin/prepare_minicif_dataset.py \
  --raw-dir "${RAW_DIR}" \
  --out-dir "${OUT_DIR}" \
  --raw-from-gzip \
  --representation minicif_v2 \
  --num-workers "${SLURM_CPUS_PER_TASK:-1}" \
  "$@"
