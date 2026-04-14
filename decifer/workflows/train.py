#!/usr/bin/env python3

from decifer.training import (
    RandomBatchSampler,
    TrainConfig,
    main,
    parse_config,
    resolve_run_dir,
    run_training,
    setup_datasets,
)


if __name__ == "__main__":
    main()
