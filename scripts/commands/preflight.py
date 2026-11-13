#!/usr/bin/env python3
"""Validate the immutable plan and the configured training environment."""

import argparse
import json
from pathlib import Path

from scripts.preflight_checks.pipeline import check as check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--api-smoke", action="store_true")
    args = parser.parse_args()

    _, _, report = check(args.config, args.api_smoke)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
