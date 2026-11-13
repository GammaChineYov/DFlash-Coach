"""Run the preflight stages and record their combined result."""

import json
from pathlib import Path

from scripts.coach_core.configuration import load_config
from scripts.coach_core.files import atomic_json, digest
from scripts.mentoring.review import read_key, review
from scripts.preflight_checks.artifacts import (
    check_configuration,
    check_models,
    check_runtime,
    check_source_inputs,
)
from scripts.preflight_checks.environment import training_versions
from scripts.preflight_checks.inputs import check_shards, check_tokens
from scripts.preflight_checks.resources import check_disk_space


def check_mentor_setup(cfg, run, plan):
    evidence = {
        "phase": "setup_only",
        "training_started": False,
        "tasks": plan["splits"],
        "shards": len(plan["shards"]),
        "epoch_limit": cfg.get("max_epochs"),
    }
    decision = review(cfg["mentor"], run, "setup-smoke", evidence)
    if decision["action"] != "continue":
        raise RuntimeError("Mentor setup was not accepted")
    return decision


def check(config_path, api_smoke=False):
    cfg = load_config(config_path)
    run = Path(cfg["run_dir"])
    plan_path = run / "plan.json"
    plan = json.loads(plan_path.read_text())

    check_configuration(cfg, plan)
    check_models(cfg, plan)
    check_source_inputs(cfg, plan)
    check_runtime(plan)
    check_shards(run, plan)
    check_tokens(run, plan)
    free, required = check_disk_space(cfg, run)
    read_key(cfg["mentor"])
    versions = training_versions(cfg)

    report = {
        "status": "cpu_preflight_passed",
        "free_gib": round(free / 2**30, 2),
        "required_gib": round(required / 2**30, 2),
        "versions": versions,
        "gpu_training_tested": False,
        "capture_tested": False,
        "plan_sha256": digest(plan_path),
    }
    if api_smoke:
        report["mentor_setup"] = check_mentor_setup(cfg, run, plan)
        report["mentor_api_tested"] = True

    atomic_json(run / "evidence/preflight.json", report)
    return cfg, plan, report
