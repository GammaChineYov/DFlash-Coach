"""Translate evidence and provider responses to the mentor JSON protocol."""

import json
import urllib.request

from scripts.coach_core.constants.mentor import SYSTEM
from scripts.mentoring.decisions import validate_decision


def review_body(cfg, evidence):
    setup_only = evidence.get("phase") == "setup_only"
    thinking = "disabled" if setup_only else "enabled"
    reasoning_effort = "none" if setup_only else cfg.get("reasoning_effort", "high")

    return {
        "model": cfg["model"],
        "messages": [
            {
                "role": "system",
                "content": SYSTEM + "\nReview language: " + cfg.get("language", "en"),
            },
            {
                "role": "user",
                "content": json.dumps(evidence, ensure_ascii=False),
            },
        ],
        "thinking": {"type": thinking},
        "reasoning_effort": reasoning_effort,
        "response_format": {"type": "json_object"},
        "stream": False,
    }


def review_request(cfg, body, credential):
    return urllib.request.Request(
        cfg["endpoint"],
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + credential,
        },
    )


def response_decision(result):
    choice = result["choices"][0]
    if choice["finish_reason"] != "stop":
        raise ValueError("Mentor response incomplete")

    content = json.loads(choice["message"]["content"])
    return validate_decision(content)
