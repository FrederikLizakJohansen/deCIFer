#!/bin/bash
#SBATCH --time 3-00:00:00
#SBATCH --job-name=prepare_data
#SBATCH --array 0
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=24G
#SBATCH --output=logs/prepare_dataset_%A_%a.out

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
ARGS=("$@")

# Display the arguments
echo "Arguments passed: ${ARGS[*]}"

python bin_refactored/prepare_dataset.py "${ARGS[@]}"
