#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --job-name=pxrd_encoder_pretrain
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/pxrd_encoder_pretrain_%j.out

set -euo pipefail

mkdir -p logs
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

if [ "$#" -eq 0 ]; then
  set -- --config configs/minicif_pxrd_encoder_pretrain_hybrid.yaml
fi

echo "Running on host: $(hostname)"
echo "Started at: $(date)"
echo "Arguments passed: $*"

python bin/pretrain_pxrd_encoder.py "$@"

echo "Finished at: $(date)"
