"""Estimate the remaining disk space needed by one training run."""

import os
from pathlib import Path

from scripts.coach_core.constants.recipe import FEATURE_BYTES_PER_TOKEN
from scripts.coach_core.files import read_rows


def monitor_bytes(cfg, run):
    total = 0
    for row in read_rows(run / "data/monitor.jsonl"):
        tokens = row["prompt_tokens"] + cfg["max_new_tokens"] + 1
        total += tokens * FEATURE_BYTES_PER_TOKEN
    return total


def allocated_bytes(run):
    total = 0
    for directory in ("features", "shards", "outputs"):
        for path in (run / directory).rglob("*"):
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
    return total


def check_disk_space(cfg, run):
    filesystem = os.statvfs(run)
    free = filesystem.f_bavail * filesystem.f_frsize

    checkpoint = int(cfg.get("checkpoint_bytes_estimate", 7500000000))
    shard_budget = int(cfg["shard_feature_budget_gib"] * 2**30)
    draft_bytes = Path(cfg["initial_draft"]).stat().st_size
    required_total = monitor_bytes(cfg, run) + shard_budget
    required_total += 2 * checkpoint + draft_bytes
    required = max(0, required_total - allocated_bytes(run))

    if free < required:
        raise RuntimeError(
            f"Need {required / 2**30:.1f} GiB free; "
            f"available {free / 2**30:.1f}"
        )
    return free, required
