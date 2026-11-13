"""Numerical validity checks; quality decisions remain with the mentor."""

import math
import statistics
from collections.abc import Mapping

from scripts.coach_core.constants.recipe import METRICS
from scripts.coach_core.types import Record


def weighted_metrics(records: list[Record]) -> Record:
    sample_count = sum(record["samples"] for record in records)
    result = {}
    for key in records[0]:
        if key == "samples":
            continue
        total = sum(record[key] * record["samples"] for record in records)
        result[key] = total / sample_count
    return result


def metric_guard(baseline: Mapping[str, float], current: Mapping[str, float]) -> None:
    # Quality judgments belong to the mentor; local code rejects invalid numerics only.
    if any(not math.isfinite(float(x)) for x in current.values()):
        raise ValueError("Non-finite validation metric")


def trend_summary(history: list[Record], best_step: int) -> Record:
    if not history:
        return {"checkpoints": 0, "steps_without_new_best": 0, "shards_without_new_best": 0}

    def window(rows: list[Record]) -> Record | None:
        if not rows:
            return None
        return {
            "count": len(rows),
            "first_step": rows[0]["steps"],
            "last_step": rows[-1]["steps"],
            "mean": {
                key: statistics.mean(r["metrics"][key] for r in rows)
                for key in METRICS
                if all(key in r["metrics"] for r in rows)
            },
        }

    result: Record = {
        "checkpoints": len(history),
        "best_step": best_step,
        "steps_without_new_best": max(0, history[-1]["steps"] - best_step),
        "shards_without_new_best": sum(r["steps"] > best_step for r in history),
        "previous_10": window(history[-20:-10]),
        "latest_10": window(history[-10:]),
        "monitor_reused_for_checkpoint_selection": True,
        "independent_test_evaluation_available": False,
    }
    rate = history[-1].get("learning_rate")
    if rate is not None:
        first = len(history) - 1
        while first > 0 and history[first - 1].get("learning_rate") == rate:
            first -= 1
        result.update(
            current_learning_rate=rate,
            shards_at_current_learning_rate=len(history) - first,
            since_latest_learning_rate_change=window(history[first:]),
        )
    return result
