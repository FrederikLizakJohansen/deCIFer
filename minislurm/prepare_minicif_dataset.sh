#!/bin/bash
#SBATCH --time 1-00:00:00
#SBATCH --job-name=prepare_minicif
#SBATCH --array 0
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/minicif_prepare_%A_%a.out

usage() {
  echo "Usage: sbatch $0 --raw-dir RAW_DIR --out-dir OUT_DIR [prepare_minicif_dataset.py options]"
  echo "Example: sbatch $0 --raw-dir data/noma --out-dir data/minicif --raw-from-gzip --num-workers 8"
  exit 1
}

if [ "$#" -eq 0 ]; then
  usage
fi

ARGS=("$@")

mkdir -p logs
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "Arguments passed: ${ARGS[*]}"
echo "Running on host: $(hostname)"
echo "Started at: $(date)"

python bin/prepare_minicif_dataset.py "${ARGS[@]}"

echo "Finished at: $(date)"
