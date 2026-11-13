"""Persist cached decisions, provider usage and retry evidence."""

import hashlib
import json
import time
from pathlib import Path

from scripts.coach_core.files import atomic_json
from scripts.mentoring.decisions import validate_decision


class ReviewStore:
    def __init__(self, run, key):
        evidence = Path(run) / "evidence"
        self.key = key
        self.output = evidence / "mentor" / (key + ".json")
        self.usage_path = evidence / "mentor-usage.json"
        self.waiting_path = evidence / "mentor-waiting.json"

    def cached_decision(self, evidence):
        if not self.output.exists():
            return None

        previous = json.loads(self.output.read_text())
        if previous.get("evidence") == evidence:
            try:
                return validate_decision(previous["decision"])
            except ValueError:
                pass

        previous_evidence = json.dumps(previous.get("evidence"), sort_keys=True)
        evidence_hash = hashlib.sha256(previous_evidence.encode()).hexdigest()[:16]
        archive = self.output.with_name(
            self.key + "-previous-" + evidence_hash + ".json"
        )
        self.output.replace(archive)
        return None

    def load_usage(self):
        if self.usage_path.exists():
            return json.loads(self.usage_path.read_text())

        return {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def record_attempt(self, ledger):
        ledger["requests"] += 1
        atomic_json(self.usage_path, ledger)

    def record_usage(self, ledger, usage):
        ledger["prompt_tokens"] += usage.get("prompt_tokens", 0)
        ledger["completion_tokens"] += usage.get("completion_tokens", 0)
        atomic_json(self.usage_path, ledger)

    def save_result(self, cfg, result, decision, evidence, usage):
        record = {
            "decision": decision,
            "model_requested": cfg["model"],
            "model_returned": result.get("model"),
            "usage": usage,
            "request_id": result.get("id"),
            "evidence": evidence,
            "time_unix": time.time(),
        }
        atomic_json(self.output, record)

    def record_waiting(self, attempt, reason):
        record = {
            "key": self.key,
            "attempt": attempt,
            "reason": reason,
            "time_unix": time.time(),
            "training_waiting": True,
        }
        atomic_json(self.waiting_path, record)
