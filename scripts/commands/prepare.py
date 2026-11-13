#!/usr/bin/env python3
"""Prepare a new immutable capture plan."""

import argparse
import json
from pathlib import Path

from scripts.coach_core.configuration import load_config
from scripts.preparation.pipeline import prepare_run
from scripts.preparation.records import normalize as normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    summary = prepare_run(config)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
