"""Validate mentor decisions and apply their permitted state changes."""

import math


def validate_decision(value):
    if not isinstance(value, dict) or set(value) != {"action", "reason", "learning_rate"}:
        raise ValueError("Mentor returned an invalid decision schema")
    if value["action"] not in ("continue", "stop"):
        raise ValueError("Unsupported mentor action")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("Invalid mentor reason")
    rate = value["learning_rate"]
    if rate is not None and (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(rate)
        or not math.isfinite(rate * 3)
        or rate <= 0
    ):
        raise ValueError("Learning rate must be positive and finite")
    return value


def apply_decision(state, decision):
    decision = validate_decision(decision)
    if decision["learning_rate"] is not None:
        state["learning_rate"] = decision["learning_rate"]
    return decision["action"] == "continue"
