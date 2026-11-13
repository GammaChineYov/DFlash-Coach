"""Normalize a validated record without changing its original payload."""

from scripts.coach_core.constants.datasets import COMPACT_PROMPT_LIMIT
from scripts.preparation.validation import validate_record


def length_bucket(prompt_tokens):
    if prompt_tokens <= COMPACT_PROMPT_LIMIT:
        return "compact"
    return "extended"


def normalize(row, split):
    row = dict(row)
    validate_record(row, split)

    row["split"] = split
    row["dataset_split"] = split
    row.setdefault("source_group", row["project"] + ":" + row["source_path"])
    row.setdefault("length_bucket", length_bucket(row["prompt_tokens"]))
    return row
