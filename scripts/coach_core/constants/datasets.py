"""The supported input splits and normalized record fields."""

import re

DATASET_SPLITS = ("train", "validation", "test")
REQUIRED_INPUT_FIELDS = (
    "project",
    "source_path",
    "source_sha256",
    "prompt",
    "language",
    "sampling_group",
)
SAMPLE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
SOURCE_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
COMPACT_PROMPT_LIMIT = 1536
