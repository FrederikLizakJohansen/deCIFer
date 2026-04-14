#!/usr/bin/env python3

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from decifer.workflows import train as workflow

RandomBatchSampler = workflow.RandomBatchSampler
TrainConfig = workflow.TrainConfig
parse_config = workflow.parse_config
setup_datasets = workflow.setup_datasets
main = workflow.main


if __name__ == "__main__":
    main()
