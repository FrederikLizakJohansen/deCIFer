#!/bin/bash
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time 1-00:00:00
#SBATCH --job-name=eval_minicif
#SBATCH --array 0
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --output=logs/minicif_eval_%A_%a.out

DEFAULT_CHECKPOINT="minicif_model_small/ckpt.pt"
DEFAULT_DATASET_DIR="data"

usage() {
  echo "Usage: sbatch $0 [visualize_minicif.py options]"
  echo "Default: sbatch $0 --checkpoint ${DEFAULT_CHECKPOINT} --dataset-dir ${DEFAULT_DATASET_DIR}"
  echo "Example: sbatch $0 --checkpoint minicif_model_small/ckpt.pt --dataset-dir data --splits val test --num-reps 8"
  exit 1
}

ARGS=("$@")
if [ "$#" -eq 0 ]; then
  ARGS=(--checkpoint "${DEFAULT_CHECKPOINT}" --dataset-dir "${DEFAULT_DATASET_DIR}")
fi

for arg in "${ARGS[@]}"; do
  if [ "${arg}" = "--help" ] || [ "${arg}" = "-h" ]; then
    usage
  fi
done

mkdir -p logs
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "Arguments passed: ${ARGS[*]}"
echo "Running on host: $(hostname)"
echo "Started at: $(date)"

python bin/visualize_minicif.py "${ARGS[@]}"

echo "Finished at: $(date)"
