#!/bin/bash
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time 2-00:00:00
#SBATCH --job-name=train_minicif
#SBATCH --array 0
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/minicif_train_%A_%a.out

DEFAULT_CONFIG="configs/minicif_small_config.yaml"

usage() {
  echo "Usage: sbatch $0 [--config CONFIG]"
  echo "Default: sbatch $0 --config ${DEFAULT_CONFIG}"
  echo "Pass any bin/train.py arguments after the script name."
  exit 1
}

ARGS=("$@")
if [ "$#" -eq 0 ]; then
  ARGS=(--config "${DEFAULT_CONFIG}")
fi

mkdir -p logs

echo "Arguments passed: ${ARGS[*]}"
echo "Running on host: $(hostname)"
echo "Started at: $(date)"

python bin/train.py "${ARGS[@]}"

echo "Finished at: $(date)"
