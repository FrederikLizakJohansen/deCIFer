#!/bin/bash
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time 1-00:00:00
#SBATCH --job-name=eval_mcif_v2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --output=logs/minicif_v2_eval_%j.out

CHECKPOINT="${CHECKPOINT:-models/minicif_v2/peak/standard/medium/ckpt.pt}"
DATASET_DIR="${DATASET_DIR:-data/noma_minicif_v2}"
OUT_DIR="${OUT_DIR:-models/minicif_v2/peak/standard/medium/minicif_report}"

mkdir -p logs
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

python bin/visualize_minicif.py \
  --checkpoint "${CHECKPOINT}" \
  --dataset-dir "${DATASET_DIR}" \
  --out-dir "${OUT_DIR}" \
  --splits val test \
  --prompt-modes pxrd pxrd-elements pxrd-stoichiometry \
                 pxrd-stoichiometry-cs pxrd-stoichiometry-cs-sg \
  --num-reps 8 \
  --generation-batch-size 8 \
  "$@"
