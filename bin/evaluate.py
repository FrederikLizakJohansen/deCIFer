#!/usr/bin/env python3

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from decifer.workflows import evaluate as workflow

build_arg_parser = workflow.build_arg_parser
evaluation_filename = workflow.evaluation_filename
main = workflow.main
parse_config = workflow.parse_config
save_evaluation = workflow.save_evaluation
worker = workflow.worker
DeciferDataset = workflow.DeciferDataset


def process_dataset(*args, **kwargs):
    original_dataset_cls = workflow.DeciferDataset
    workflow.DeciferDataset = DeciferDataset
    try:
        return workflow.process_dataset(*args, **kwargs)
    finally:
        workflow.DeciferDataset = original_dataset_cls


if __name__ == "__main__":
    main()
