"""Write normalized run inputs and bounded training shards."""

import hashlib
import math

from scripts.coach_core.constants.recipe import FEATURE_BYTES_PER_TOKEN
from scripts.coach_core.files import digest, write_rows
from scripts.coach_core.planning import shard_rows


def write_inputs(run, dataset):
    train = dataset.train
    monitor = dataset.monitor
    benchmark = dataset.benchmark
    groups = dataset.splits
    for name in ("data", "evidence", "outputs", "features", "shards", "configs"):
        (run / name).mkdir(parents=True, exist_ok=True)
    for name, rows in [
        ("train-all", train),
        ("monitor", monitor),
        ("benchmark-test", benchmark),
        ("validation-all", groups["validation"]),
        ("test-all", groups["test"]),
        ("tokenize-all", train + monitor + benchmark),
    ]:
        write_rows(run / "data" / (name + ".jsonl"), trainer_rows(rows))


def write_shards(cfg, run, train):
    def training_order(row):
        identity = "train:" + row["id"]
        return hashlib.sha256(identity.encode()).hexdigest()

    ordered = sorted(train, key=training_order)
    shards = []
    batches = shard_rows(
        ordered,
        cfg["max_shard_rows"],
        int(cfg["shard_feature_budget_gib"] * 2**30),
        cfg["max_new_tokens"],
    )
    for index, rows in enumerate(batches):
        key = f"e00-s{index:04d}"
        path = run / "data/shards" / (key + ".jsonl")
        write_rows(path, rows)
        shards.append(
            {
                "key": key,
                "path": str(path),
                "sha256": digest(path),
                "rows": len(rows),
                "steps": math.ceil(len(rows) / cfg["batch_size"]),
                "feature_bytes_upper_bound": sum(
                    (row["prompt_tokens"] + cfg["max_new_tokens"] + 1)
                    * FEATURE_BYTES_PER_TOKEN
                    for row in rows
                ),
            }
        )
    return shards


def trainer_rows(rows):
    normalized = []
    for row in rows:
        sample = dict(row)
        split = row["dataset_split"]
        sample["split"] = "eval" if split == "validation" else split
        normalized.append(sample)
    return normalized
