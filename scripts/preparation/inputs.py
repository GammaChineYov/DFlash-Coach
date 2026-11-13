"""Load isolated dataset splits and select fixed validation/test holdouts."""

import hashlib
from pathlib import Path

from scripts.coach_core.constants.datasets import DATASET_SPLITS
from scripts.coach_core.files import read_rows
from scripts.coach_core.planning import validate_splits
from scripts.coach_core.types.preparation import PreparedDataset
from scripts.preparation.records import normalize


def read_splits(source):
    groups = {}
    for split in DATASET_SPLITS:
        path = source / (split + ".jsonl")
        groups[split] = [normalize(row, split) for row in read_rows(path)]
    return groups


def check_source_overlap(tasks):
    seen = {}
    for row in tasks:
        prompt_hash = hashlib.sha256(row["prompt"].encode()).hexdigest()
        identities = (
            ("prompt", prompt_hash),
            ("source", row["source_sha256"]),
        )
        for identity in identities:
            previous_split = seen.setdefault(identity, row["split"])
            if previous_split != row["split"]:
                raise ValueError("A prompt or source hash crosses splits")


def choose_holdout(rows, split, count):
    def monitor_order(row):
        identity = "monitor:" + row["id"]
        return hashlib.sha256(identity.encode()).hexdigest()

    ordered = sorted(rows, key=monitor_order)
    if len(ordered) < count:
        raise ValueError(f"{split} needs at least {count} records")
    return ordered[:count]


def load_dataset(cfg):
    groups = read_splits(Path(cfg["dataset"]))
    tasks = []
    for rows in groups.values():
        tasks.extend(rows)

    counts = validate_splits(tasks)
    if any(not rows for rows in groups.values()):
        raise ValueError("All three splits must be nonempty")
    check_source_overlap(tasks)

    monitor = choose_holdout(
        groups["validation"], "validation", cfg["validation_monitor_rows"],
    )
    benchmark = choose_holdout(
        groups["test"], "test", cfg["benchmark_test_rows"],
    )
    if cfg["validation_monitor_rows"] < 16:
        raise ValueError("The trainer requires at least 16 validation records")

    selected = groups["train"] + monitor + benchmark
    maximum = max(row["prompt_tokens"] for row in selected)
    required_context = maximum + cfg["max_new_tokens"] + 1
    if required_context > cfg["capture_context"]:
        raise ValueError("capture_context is too small")

    return PreparedDataset(
        splits=groups,
        counts=counts,
        train=groups["train"],
        monitor=monitor,
        benchmark=benchmark,
        max_prompt_tokens=maximum,
    )
