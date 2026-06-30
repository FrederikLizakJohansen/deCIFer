#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --job-name=minicif_ablation
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/minicif_ablation_%j.out

set -euo pipefail

mkdir -p logs
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "Running on host: $(hostname)"
echo "Started at: $(date)"
echo "Arguments passed: $*"

python bin/run_minicif_condition_ablation.py "$@"

echo "Finished at: $(date)"
