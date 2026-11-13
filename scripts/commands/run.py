#!/usr/bin/env python3
"""Launch or resume an explicitly authorized training run."""

import argparse
from pathlib import Path

from scripts.orchestration.controller import start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--start-authorized", action="store_true")
    parser.add_argument("--restore-only", action="store_true")
    args = parser.parse_args()

    start(
        config_path=args.config,
        start_authorized=args.start_authorized,
        restore_only=args.restore_only,
    )


if __name__ == "__main__":
    main()
