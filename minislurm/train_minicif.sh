#!/bin/bash
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time 2-00:00:00
#SBATCH --job-name=train_minicif
#SBATCH --array 0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/minicif_train_%A_%a.out

DEFAULT_CONFIG="configs/minicif_small_config.yaml"

usage() {
  echo "Usage: sbatch $0 [--config CONFIG]"
  echo "Default: sbatch $0 --config ${DEFAULT_CONFIG}"
  echo "Multi-GPU example: sbatch --gres=gpu:a100:4 $0 --config configs/minicif_large_config.yaml"
  echo "Pass any bin/train.py arguments after the script name."
  exit 1
}

ARGS=("$@")
if [ "$#" -eq 0 ]; then
  ARGS=(--config "${DEFAULT_CONFIG}")
fi

mkdir -p logs
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "Arguments passed: ${ARGS[*]}"
echo "Running on host: $(hostname)"
echo "Started at: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-1}"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]; then
  GPUS_PER_NODE=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
fi
if ! [[ "${GPUS_PER_NODE}" =~ ^[0-9]+$ ]]; then
  GPUS_PER_NODE=1
fi

echo "GPUs visible to job: ${GPUS_PER_NODE}"
if [ -z "${OMP_NUM_THREADS:-}" ]; then
  OMP_THREADS=$(( ${SLURM_CPUS_PER_TASK:-1} / GPUS_PER_NODE ))
  if [ "${OMP_THREADS}" -lt 1 ]; then
    OMP_THREADS=1
  fi
  export OMP_NUM_THREADS="${OMP_THREADS}"
fi
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"

if [ "${GPUS_PER_NODE}" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}" bin/train.py "${ARGS[@]}"
else
  python bin/train.py "${ARGS[@]}"
fi

echo "Finished at: $(date)"
