#!/usr/bin/env python3
"""Explicitly launched, resumable shard capture/training with a mentor veto."""

import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from scripts.coach_core.configuration import load_config
from scripts.coach_core.files import atomic_json, digest
from scripts.coach_core.reports import epoch_report, startup_report
from scripts.mentoring import review as mentor
from scripts.mentoring.decisions import apply_decision
from scripts.preflight_checks.pipeline import check
from scripts.orchestration import runtime
from scripts.orchestration.shards import train_shard
from scripts.orchestration.session import (
    ensure_production_idle,
    install_interrupt_handlers,
    load_recovery_context,
    load_run_state,
    training_environment,
)
from scripts.mentoring.strategy import pending_strategy


def review_startup(cfg, plan, run, state, previous_error, recovery_context):
    state_path = run / "evidence/run-state.json"
    strategy = pending_strategy(cfg, run, state)
    if strategy:
        decision = strategy["decision"]
        state.setdefault("applied_strategy_requests", []).append(strategy["request_id"])
        state["last_strategy_review"] = strategy
    else:
        phase = "resume_training" if state["committed"] else "before_training"
        report_path = run / "outputs/training.json"
        resume_metrics = json.loads(report_path.read_text()) if report_path.exists() else None
        decision = mentor.review(
            cfg["mentor"],
            run,
            f"{phase}-e{state['epoch']}-s{state['next_shard']}",
            startup_report(
                cfg,
                plan,
                state,
                resume_metrics,
                previous_error,
                recovery_context,
            ),
        )
    if not apply_decision(state, decision):
        if not (run / "outputs/training.json").exists():
            state.update(
                status="completed",
                stage="mentor_finished_before_training",
                mentor_final=decision,
                candidate_installed=False,
                finished_unix=time.time(),
            )
            return False
        state.update(training_finished=True, training_stop_reason=decision["reason"])
    atomic_json(state_path, state)
    return True


def train_epochs(cfg, plan, run, state, env):
    state_path = run / "evidence/run-state.json"
    training_finished = state.get("training_finished", False)
    report = run / "outputs/training.json"
    metrics = json.loads(report.read_text()) if report.exists() else None
    while not training_finished:
        epoch = state["epoch"]
        # Each epoch visits the whole train corpus, in an epoch-specific shard order.
        templates = sorted(
            plan["shards"],
            key=lambda row: (
                row.get("priority", 1),
                hashlib.sha256((str(epoch) + row["key"]).encode()).hexdigest(),
            ),
        )
        for index in range(state["next_shard"], len(templates)):
            metrics = train_shard(cfg, plan, run, state, env, templates[index], index)
            if state.get("training_finished"):
                training_finished = True
                break
        if (
            not training_finished
            and cfg.get("max_epochs") is not None
            and epoch + 1 - cfg.get("initial_epoch", 0) >= cfg["max_epochs"]
        ):
            training_finished = True
            state.update(
                training_finished=True,
                training_stop_reason="Configured max_epochs reached",
            )
            atomic_json(state_path, state)
        if not training_finished:
            decision = mentor.review(
                cfg["mentor"],
                run,
                f"epoch-{epoch:04d}-complete",
                epoch_report(cfg, plan, state, metrics, epoch),
            )
            if not apply_decision(state, decision):
                state["training_stop_reason"] = decision["reason"]
                training_finished = True
                state["training_finished"] = True
            else:
                state.update(epoch=epoch + 1, next_shard=0, stage="next_epoch")
            state["last_mentor_decision"] = decision
            atomic_json(state_path, state)


def start(config_path, start_authorized=False, restore_only=False):
    cfg = load_config(config_path)
    run = Path(cfg["run_dir"])
    marker = run / "evidence/restore-needed"
    if restore_only:
        runtime.restore(cfg, marker)
        return
    if not start_authorized:
        raise SystemExit(
            "Prepared only. Full capture/training requires "
            "the explicit --start-authorized flag."
        )
    lock_path = run / "evidence/trial.lock"
    with os.fdopen(os.open(lock_path, os.O_RDONLY | os.O_CREAT, 0o600), "r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(config_path, run, marker)


def run_stages(config_path, cfg, plan, run, state, env, previous_error, recovery_context):
    state_path = run / "evidence/run-state.json"
    if not state.get("training_finished") and not review_startup(
        cfg, plan, run, state, previous_error, recovery_context
    ):
        return
    runtime.service_control("stop", cfg["production_service"], cfg.get("production_scope", "user"))
    for committed_key in state["committed"]:
        old_features = run / "shards" / committed_key / "features"
        if old_features.exists():
            shutil.rmtree(old_features)
    monitor = run / "features/monitor"
    runtime.capture(
        cfg,
        run / "data/monitor.jsonl",
        monitor,
        env,
        run / "evidence/capture-monitor.log",
    )
    train_epochs(cfg, plan, run, state, env)
    state["stage"] = "benchmark"
    atomic_json(state_path, state)
    runtime.run_stage(
        [
            cfg["python"],
            str(Path(__file__).resolve().parents[2] / "coach.py"),
            "benchmark", "--config", str(config_path),
        ],
        env,
        run / "evidence/benchmark.log",
        cfg,
    )
    summary = json.loads((run / "outputs/benchmark-summary.json").read_text())
    decision = mentor.review(
        cfg["mentor"],
        run,
        "final-benchmark",
        {"phase": "final_benchmark", **summary},
    )
    state.update(
        status="completed",
        stage="candidate_ready_for_review",
        mentor_final=decision,
        candidate_installed=False,
        finished_unix=time.time(),
    )


def execute(config_path, run, marker):
    """Run a locked trial and persist its state after restoring the service."""
    cfg, plan, _ = check(config_path)
    state_path = run / "evidence/run-state.json"
    state = load_run_state(cfg, state_path)
    if state["status"] == "completed":
        print("Run is already completed; candidate remains uninstalled.")
        return

    install_interrupt_handlers()


    ensure_production_idle(cfg)
    env = training_environment(cfg, plan)


    if cfg.get("production_service"):
        marker.write_text("restore configured inference service on exit\n")
    previous_error = state.get("error")
    recovery_context = load_recovery_context(run, state, previous_error)


    state.update(
        status="running",
        error=None,
        started_unix=state.get("started_unix") or time.time(),
    )
    atomic_json(state_path, state)
    try:
        run_stages(
            config_path, cfg, plan, run, state, env,
            previous_error, recovery_context,
        )
    except BaseException as error:
        state.update(status="stopped", error=str(error), candidate_installed=False)
        raise
    finally:
        try:
            runtime.restore(cfg, marker)
            state["production_restored"] = True
            config_path = cfg.get("production_config")
            config_unchanged = True
            if config_path:
                expected = plan["production_config_sha256"]
                config_unchanged = digest(config_path) == expected
            state["production_config_unchanged"] = config_unchanged
            if not state["production_config_unchanged"]:
                state.update(
                    status="stopped",
                    error="Production config changed during training",
                )
        finally:
            atomic_json(state_path, state)
