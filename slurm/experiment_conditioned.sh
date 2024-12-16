#!/bin/bash
#SBATCH -p gpu --gres=gpu:a100:1
# #SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time 1-12:00:00
#SBATCH --job-name=eval_deCIFer
# #SBATCH --array 0-14%15
#SBATCH --array 0-2%3
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=12G
#SBATCH --output=logs/exp_cond_%A_%a.out

# Function to display help message
usage() {
  echo "Usage: $0 [options]"
  echo "Pass any number of arguments and their values to the script, e.g. --a value1 --b value2"
  exit 1
}

# Check if any arguments are provided
if [ "$#" -eq 0 ]; then
  usage
fi

# Collect all aguments
EXTERNAL_ARGS=("$@")

# Define specific argument sets for each array job
ARGS_ARRAY=(
  # New baseline with extended Q-range
  "--dataset-name QE_none_N-0p00_B-0p05 --add-broadening 0.05"
  "--dataset-name QE_comp_N-0p00_B-0p05 --add-composition --add-broadening 0.05"
  "--dataset-name QE_compSG_N-0p00_B-0p05 --add-composition --add-spacegroup --add-broadening 0.05"

#  "--dataset-name none_N-0p00_B-0p05 --add-broadening 0.05"
#  "--dataset-name comp_N-0p00_B-0p05 --add-composition --add-broadening 0.05"
#  "--dataset-name compSG_N-0p00_B-0p05 --add-composition --add-spacegroup --add-broadening 0.05"
#
#  "--dataset-name none_N-0p05_B-0p05 --add-noise 0.05 --add-broadening 0.05"
#  "--dataset-name comp_N-0p05_B-0p05 --add-composition --add-noise 0.05 --add-broadening 0.05"
#  "--dataset-name compSG_N-0p05_B-0p05 --add-composition --add-spacegroup --add-noise 0.05 --add-broadening 0.05"
#
#  "--dataset-name none_N-0p00_B-0p10 --add-broadening 0.10"
#  "--dataset-name comp_N-0p00_B-0p10 --add-composition --add-broadening 0.10"
#  "--dataset-name compSG_N-0p00_B-0p10 --add-composition --add-spacegroup --add-broadening 0.10"
#
#  "--dataset-name none_N-0p05_B-0p10 --add-noise 0.05 --add-broadening 0.10"
#  "--dataset-name comp_N-0p05_B-0p10 --add-composition --add-noise 0.05 --add-broadening 0.10"
#  "--dataset-name compSG_N-0p05_B-0p10 --add-composition --add-spacegroup --add-noise 0.05 --add-broadening 0.10"
# 
#  # OOD
#  "--dataset-name none_N-0p10_B-0p05 --add-noise 0.10 --add-broadening 0.05"
#  "--dataset-name none_N-0p00_B-0p20 --add-broadening 0.20"
#  "--dataset-name none_N-0p10_B-0p20 --add-noise 0.10 --add-broadening 0.20"
  
#  # OOD
#  "--dataset-name comp_N-0p10_B-0p05 --add-composition --add-noise 0.10 --add-broadening 0.05"
#  "--dataset-name comp_N-0p00_B-0p20 --add-composition --add-broadening 0.20"
#  "--dataset-name comp_N-0p10_B-0p20 --add-composition --add-noise 0.10 --add-broadening 0.20"
)

ARRAY_ARGS=${ARGS_ARRAY[$SLURM_ARRAY_TASK_ID]}

# Display the arguments
echo "Model and data arguments passed: ${EXTERNAL_ARGS[*]}"
echo "Experimental arguments passed: $ARRAY_ARGS"

python bin_refactored/evaluate.py $ARRAY_ARGS "${EXTERNAL_ARGS[@]}"
