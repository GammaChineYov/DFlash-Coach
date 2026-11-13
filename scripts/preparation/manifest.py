"""Freeze hashes and provenance for every input to the training run."""

from pathlib import Path

from scripts.coach_core.configuration import command_argv
from scripts.coach_core.constants.paths import PROJECT_ROOT
from scripts.coach_core.files import digest


def code_hashes():
    files = [PROJECT_ROOT / "coach.py"]
    for directory in ("scripts", "source"):
        files.extend(sorted((PROJECT_ROOT / directory).rglob("*.py")))
    return {str(path.relative_to(PROJECT_ROOT)): digest(path) for path in files}


def binary_hashes(cfg):
    hashes = {}
    for key in ("capture_binary", "tokenize_binary"):
        path = Path(cfg[key])
        hashes[str(path)] = digest(path)
    server = command_argv(cfg)[0]
    hashes[server] = digest(server)
    return hashes


def source_hashes(cfg, splits):
    source = Path(cfg["dataset"])
    hashes = {}
    for split in splits:
        filename = split + ".jsonl"
        hashes[filename] = digest(source / filename)
    return hashes


def optional_digest(cfg, key):
    if cfg.get(key):
        return digest(cfg[key])
    return None


def build_plan(cfg, run, dataset, shards):
    plan = {
        "version": 1,
        "config": cfg,
        "splits": dataset.counts,
        "train_tasks": len(dataset.train),
        "shards": shards,
        "source_hashes": source_hashes(cfg, dataset.splits),
        "binaries": binary_hashes(cfg),
        "code_hashes": code_hashes(),
        "target_sha256": digest(cfg["target_model"]),
        "initial_draft_sha256": digest(cfg["initial_draft"]),
        "monitor_sha256": digest(run / "data/monitor.jsonl"),
        "benchmark_sha256": digest(run / "data/benchmark-test.jsonl"),
        "tokenization_rows": (
            len(dataset.train) + len(dataset.monitor) + len(dataset.benchmark)
        ),
        "max_prompt_tokens": dataset.max_prompt_tokens,
        "steps_per_epoch_estimate": sum(shard["steps"] for shard in shards),
        "production_config_sha256": optional_digest(cfg, "production_config"),
        "production_launcher_sha256": optional_digest(cfg, "production_launcher"),
    }
    if cfg.get("seed_checkpoint"):
        validate_seed_config(cfg)
        plan["seed_checkpoint_sha256"] = digest(cfg["seed_checkpoint"])
    return plan


def validate_seed_config(cfg):
    if not cfg.get("seed_plan_sha256") or type(cfg.get("seed_step")) is not int:
        raise ValueError("A seed checkpoint requires seed_plan_sha256 and seed_step")
