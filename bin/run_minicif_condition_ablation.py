#!/usr/bin/env python3

import argparse
import glob
import os
import subprocess
import sys


DEFAULT_CONFIG_DIR = "configs/minicif_condition_ablation"


def discover_configs(config_dir):
    return sorted(glob.glob(os.path.join(config_dir, "*.yaml")))


def main():
    parser = argparse.ArgumentParser(description="Run minicif conditioning ablation configs sequentially.")
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--configs", nargs="*", default=None, help="Explicit config paths. Overrides --config-dir when provided.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configs = args.configs if args.configs else discover_configs(args.config_dir)
    if not configs:
        raise FileNotFoundError(f"no .yaml configs found under {args.config_dir}")

    for config_path in configs:
        command = [args.python, "bin/train.py", "--config", config_path]
        print("Running:", " ".join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
