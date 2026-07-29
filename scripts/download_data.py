"""Standalone data-download script (mirrors ``alphaduel download``).

Usage: python scripts/download_data.py --config configs/experiment/p0_mvp.yaml
"""

from __future__ import annotations

import argparse

from alphaduel.config.loader import load_experiment_config
from alphaduel.config.schema import Secrets
from alphaduel.pipeline import download_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Download & cache AlphaDuel data.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    download_data(cfg, Secrets())
    print("Data cached.")


if __name__ == "__main__":
    main()
