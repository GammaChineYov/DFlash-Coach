"""Prepare one new immutable run from validated source records."""

from pathlib import Path

from scripts.coach_core.files import atomic_json, digest
from scripts.preparation.artifacts import write_inputs, write_shards
from scripts.preparation.inputs import load_dataset
from scripts.preparation.manifest import build_plan


def prepare_run(cfg):
    run = Path(cfg["run_dir"])
    plan_path = run / "plan.json"
    state_path = run / "evidence/run-state.json"
    if plan_path.exists() or state_path.exists():
        raise SystemExit("Run already exists; use a new run_dir or resume it")

    dataset = load_dataset(cfg)
    write_inputs(run, dataset)
    shards = write_shards(cfg, run, dataset.train)
    plan = build_plan(cfg, run, dataset, shards)
    atomic_json(plan_path, plan)

    summary = {
        "status": "prepared_not_started",
        "splits": dataset.counts,
        "shards": len(shards),
        "max_prompt_tokens": dataset.max_prompt_tokens,
    }
    prepared = {
        "status": summary["status"],
        "train_tasks": len(dataset.train),
        "shards": summary["shards"],
        "max_prompt_tokens": summary["max_prompt_tokens"],
        "plan_sha256": digest(plan_path),
    }
    atomic_json(run / "evidence/prepared.json", prepared)
    return summary
