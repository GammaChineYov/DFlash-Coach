"""Validate normalized input fields before any run artifacts are written."""

from scripts.coach_core.constants.datasets import (
    REQUIRED_INPUT_FIELDS,
    SAMPLE_ID_PATTERN,
    SOURCE_HASH_PATTERN,
)


def validate_record(row, split):
    validate_id(row["id"])
    validate_split(row, split)
    validate_required_fields(row)
    validate_source_hash(row["source_sha256"])
    validate_prompt_tokens(row.get("prompt_tokens"))


def validate_id(sample_id):
    if not SAMPLE_ID_PATTERN.fullmatch(sample_id):
        raise ValueError(
            "Sample IDs must contain only letters, digits, underscores and hyphens"
        )


def validate_split(row, split):
    recorded_split = row.get("dataset_split")
    if "dataset_split" not in row:
        recorded_split = row.get("split", split)

    allowed = (split,)
    if split == "validation":
        allowed = ("validation", "eval")

    if recorded_split not in allowed:
        raise ValueError("A record has the wrong split")


def validate_required_fields(row):
    for name in REQUIRED_INPUT_FIELDS:
        value = row.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError("Missing string field: " + name)


def validate_source_hash(source_hash):
    if not SOURCE_HASH_PATTERN.fullmatch(source_hash):
        raise ValueError("Invalid source_sha256")


def validate_prompt_tokens(tokens):
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("Invalid prompt_tokens")
