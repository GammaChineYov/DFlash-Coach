"""Inputs selected for one immutable capture plan."""

from dataclasses import dataclass

from scripts.coach_core.types import Record


@dataclass
class PreparedDataset:
    splits: dict[str, list[Record]]
    counts: dict[str, int]
    train: list[Record]
    monitor: list[Record]
    benchmark: list[Record]
    max_prompt_tokens: int
