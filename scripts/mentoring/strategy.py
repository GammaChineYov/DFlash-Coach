"""Summarize validation drift and apply a provider-backed strategy review once."""

import json
import re
from pathlib import Path

from scripts.coach_core.files import digest
from scripts.coach_core.metrics import trend_summary as trend_summary
from scripts.mentoring.review import validate_decision


def pending_strategy(cfg, run, state):
    run = Path(run)
    pointer = run / "evidence/strategy-control.json"
    if not pointer.exists():
        return None
    control = json.loads(pointer.read_text())
    key = control["review_key"]
    if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", key):
        raise ValueError("Invalid strategy review key")
    if control["plan_sha256"] != digest(run / "plan.json"):
        raise ValueError("Strategy review refers to a different data plan")
    record = json.loads((run / "evidence/mentor" / (key + ".json")).read_text())
    request_id = record.get("request_id")
    if not request_id or record.get("evidence", {}).get("phase") != "strategy_review":
        raise ValueError("Strategy review has no provider request or strategy evidence")
    if record.get("model_requested") != cfg["mentor"]["model"]:
        raise ValueError("Strategy review provider does not match the configured mentor")
    if request_id in state.get("applied_strategy_requests", []):
        return None
    return {
        "request_id": request_id,
        "review_key": key,
        "decision": validate_decision(record["decision"]),
        "source_step": record["evidence"].get("current_step"),
    }
