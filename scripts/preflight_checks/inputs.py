"""Check the frozen shards, holdouts and exact tokenizer report."""

import json

from scripts.preflight_checks.artifacts import require_digest


def check_shards(run, plan):
    for shard in plan["shards"]:
        require_digest(shard["path"], shard["sha256"], "Shard changed")

    holdouts = (
        ("monitor", "monitor_sha256"),
        ("benchmark-test", "benchmark_sha256"),
    )
    for name, key in holdouts:
        path = run / "data" / (name + ".jsonl")
        require_digest(path, plan[key], "Holdout changed")


def check_tokens(run, plan):
    report_path = run / "evidence/token-verification.json"
    report = json.loads(report_path.read_text())
    has_invalid_tokens = report["invalid"] or report["mismatches"]
    has_wrong_row_count = report["rows"] != plan["tokenization_rows"]

    if has_invalid_tokens or has_wrong_row_count:
        raise ValueError("Prompt token verification failed")
