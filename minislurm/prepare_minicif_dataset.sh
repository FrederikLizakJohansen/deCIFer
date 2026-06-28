#!/bin/bash
#SBATCH --time 1-00:00:00
#SBATCH --job-name=prepare_minicif
#SBATCH --array=0-0
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/minicif_prepare_%A_%a.out

usage() {
  echo "Usage: sbatch $0 --raw-dir RAW_DIR --out-dir OUT_DIR [prepare_minicif_dataset.py options]"
  echo "Example: sbatch --array=0-31 $0 --raw-dir data/noma --out-dir data --raw-from-gzip"
  echo "Merge:   sbatch $0 --raw-dir data/noma --out-dir data --raw-from-gzip --num-shards 32 --merge-shards"
  exit 1
}

has_arg() {
  local needle="$1"
  shift
  for arg in "$@"; do
    case "$arg" in
      "$needle"|"$needle"=*) return 0 ;;
    esac
  done
  return 1
}

if [ "$#" -eq 0 ]; then
  usage
fi

ARGS=("$@")
MERGE_SHARDS=0
if has_arg "--merge-shards" "${ARGS[@]}"; then
  MERGE_SHARDS=1
fi

if ! has_arg "--num-workers" "${ARGS[@]}"; then
  ARGS+=(--num-workers "${SLURM_CPUS_PER_TASK:-$(nproc)}")
fi

if ! has_arg "--chunksize" "${ARGS[@]}"; then
  ARGS+=(--chunksize 8)
fi

if [ "$MERGE_SHARDS" -eq 0 ] && [ "${SLURM_ARRAY_TASK_COUNT:-1}" -gt 1 ]; then
  if ! has_arg "--num-shards" "${ARGS[@]}"; then
    ARGS+=(--num-shards "${SLURM_ARRAY_TASK_COUNT}")
  fi
  if ! has_arg "--shard-index" "${ARGS[@]}"; then
    ARGS+=(--shard-index "${SLURM_ARRAY_TASK_ID}")
  fi
fi

mkdir -p logs
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "Arguments passed: ${ARGS[*]}"
echo "Running on host: $(hostname)"
echo "Started at: $(date)"

python bin/prepare_minicif_dataset.py "${ARGS[@]}"

echo "Finished at: $(date)"
