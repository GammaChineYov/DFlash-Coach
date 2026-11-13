"""Bound feature storage and keep related sources within one data split."""

from collections.abc import Iterable, Iterator

from scripts.coach_core.constants.recipe import FEATURE_BYTES_PER_TOKEN
from scripts.coach_core.types import Record, SourceKey


def shard_rows(
    rows: Iterable[Record],
    max_rows: int,
    budget_bytes: int,
    max_new: int,
) -> Iterator[list[Record]]:
    shard: list[Record] = []
    used = 0
    for row in rows:
        size = (row["prompt_tokens"] + max_new + 1) * FEATURE_BYTES_PER_TOKEN
        if size > budget_bytes:
            raise ValueError("Single example exceeds shard feature budget")
        if shard and (len(shard) >= max_rows or used + size > budget_bytes):
            yield shard
            shard, used = [], 0
        shard.append(row)
        used += size
    if shard:
        yield shard


def validate_splits(tasks: Iterable[Record]) -> dict[str, int]:
    ids: set[str] = set()
    families: dict[SourceKey, str] = {}
    sources: dict[SourceKey, str] = {}
    counts: dict[str, int] = {}
    for row in tasks:
        if row["id"] in ids:
            raise ValueError("Duplicate task id")
        ids.add(row["id"])
        split = row["split"]
        if split not in ("train", "validation", "test"):
            raise ValueError("Unexpected split")
        counts[split] = counts.get(split, 0) + 1
        identities = (
            (families, row["source_group"]),
            (sources, (row["project"], row["source_path"])),
        )
        for table, key in identities:
            previous = table.setdefault(key, split)
            if previous != split:
                raise ValueError("Related source occurs across dataset splits")
    return counts
