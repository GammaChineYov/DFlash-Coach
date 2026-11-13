"""Compare speed and output consistency across measured variants."""

import statistics
from collections import defaultdict


def matching_outputs(records, reference):
    matches = 0
    for key, row in records.items():
        matches += row["output_sha256"] == reference[key]["output_sha256"]
    return matches


def median_speed_by_language(rows):
    medians = {}
    for language in ("ru", "en"):
        speeds = [
            row["timings"]["predicted_per_second"]
            for row in rows
            if row["language"] == language
        ]
        medians[language] = statistics.median(speeds)
    return medians


def variant_summary(records, references):
    rows = list(records.values())
    tokens = sum(row["timings"]["predicted_n"] for row in rows)
    decode_seconds = sum(row["timings"]["predicted_ms"] for row in rows) / 1000
    wall_seconds = sum(row["wall_seconds"] for row in rows)
    decode_tps = tokens / decode_seconds if decode_seconds else 0

    return {
        "requests": len(rows),
        "new_tokens": tokens,
        "decode_tps": decode_tps,
        "wall_tps": tokens / wall_seconds,
        "same_as_parent": matching_outputs(records, references["parent"]),
        "same_as_no_draft": matching_outputs(records, references["no_draft"]),
        "median_tps_by_language": median_speed_by_language(rows),
    }


def summarize(rows, measurements):
    by_variant = defaultdict(dict)
    for measurement in measurements:
        key = (measurement["id"], measurement["repeat"])
        by_variant[measurement["variant"]][key] = measurement

    variants = {}
    for label, records in by_variant.items():
        variants[label] = variant_summary(records, by_variant)

    parent = variants["parent"]
    candidate = variants["candidate"]
    speedup = (candidate["decode_tps"] / parent["decode_tps"] - 1) * 100
    same_outputs = candidate["same_as_parent"] == candidate["requests"]
    faster_or_equal = candidate["decode_tps"] >= parent["decode_tps"]

    return {
        "test_tasks": len(rows),
        "repeats": 2,
        "max_new_tokens": 512,
        "variants": variants,
        "candidate_installed": False,
        "longest_test_prompt_tokens": max(row["prompt_tokens"] for row in rows),
        "context_note": (
            "This benchmark does not validate the full 262144-token window."
        ),
        "decode_speedup_vs_parent_percent": speedup,
        "candidate_eligible_for_manual_review": same_outputs and faster_or_equal,
    }
