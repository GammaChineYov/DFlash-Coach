"""Resources and state passed between the training stages."""

from dataclasses import dataclass
from typing import Any

from scripts.coach_core.types import Record


@dataclass
class TrainingBackend:
    model: Any
    reader: Any
    objective: Any
    optimizer: Any
    scaler: Any


@dataclass
class TrainingProgress:
    first_step: int
    best_step: int
    best_metrics: Record
    best_model: Record
    baseline: Record
    bad: int


@dataclass
class BatchResult:
    step: int
    history: list[Record]
    overflow_retries: int
    accepted_ids: list[str]


@dataclass
class ObjectiveBatch:
    features: Any
    noise: Any
    positions: Any
    attention: Any
    gold: Any
    previous: Any
