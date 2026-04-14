#!/usr/bin/env python3

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from decifer.workflows.ablation import (
    build_arg_parser,
    experiment,
    main,
    parse_config,
)


if __name__ == "__main__":
    main()
