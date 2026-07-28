#!/bin/bash
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time 2-00:00:00
#SBATCH --job-name=train_mcif_v2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/minicif_v2_train_%j.out

CONFIG="${CONFIG:-configs/config_v2/peak/standard/medium.yaml}"
ARGS=("$@")
HAS_CONFIG=0
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --config)
      if ((i + 1 >= ${#ARGS[@]})); then
        echo "--config requires a path" >&2
        exit 2
      fi
      CONFIG="${ARGS[$((i + 1))]}"
      HAS_CONFIG=1
      ;;
    --config=*)
      CONFIG="${ARGS[$i]#--config=}"
      HAS_CONFIG=1
      ;;
  esac
done
if [ "${HAS_CONFIG}" -eq 0 ]; then
  ARGS=(--config "${CONFIG}" "${ARGS[@]}")
fi

mkdir -p logs
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  python bin/audit_minicif_v2.py \
    --config "${CONFIG}" \
    --max-items "${AUDIT_MAX_ITEMS:-100}"
fi

python bin/train.py "${ARGS[@]}"
