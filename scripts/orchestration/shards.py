"""Capture, train and commit exactly one shard after its mentor review."""

import json
import shutil
import time

from scripts.coach_core.commands import training_command
from scripts.coach_core.files import atomic_json, read_rows, write_rows
from scripts.coach_core.metrics import metric_guard
from scripts.coach_core.reports import shard_report
from scripts.mentoring import review as mentor
from scripts.mentoring.decisions import apply_decision
from scripts.orchestration import runtime


def train_shard(cfg, plan, run, state, env, template, index):
    state_path = run / "evidence/run-state.json"
    epoch = state["epoch"]
    shard = {**template, "key": f"e{epoch:04d}-" + template["key"].split("-")[-1]}
    work = run / "shards" / shard["key"]
    rows, capture_seconds, shard_started = capture_rows(
        cfg, run, work, shard, state, env,
    )
    metrics = train_rows(cfg, plan, run, work, shard, state, env)
    point = {
        "shard": shard["key"],
        "epoch": epoch + 1,
        "steps": metrics["steps"],
        "learning_rate": metrics["learning_rate"],
        "metrics": metrics["last"],
        "capture_seconds": capture_seconds,
        "shard_seconds_before_mentor": time.time() - shard_started,
        "amp_overflow_retries": metrics.get("amp_overflow_retries", 0),
    }
    decision = mentor.review(
        cfg["mentor"],
        run,
        shard["key"],
        shard_report(cfg, plan, state, shard, index, metrics, rows, point, epoch),
    )
    commit_coverage(state, state_path, work, index, point, decision)

    return metrics


def attach_monitor(run, work, rows):
    monitor = run / "features/monitor"
    val = list(read_rows(run / "data/monitor.jsonl"))
    write_rows(work / "data/all.jsonl", rows + val)
    for row in val:
        for suffix in (".json", ".f16"):
            dest = work / "features" / (row["id"] + suffix)
            if not dest.exists():
                dest.symlink_to(monitor / dest.name)


def commit_coverage(state, state_path, work, index, point, decision):
    keep_training = apply_decision(state, decision)
    state["history"].append(point)
    state["committed"].append(point["shard"])
    state.update(
        next_shard=index + 1,
        stage="checkpoint_committed",
        last_mentor_decision=decision,
    )
    state["updated_unix"] = time.time()
    atomic_json(state_path, state)
    shutil.rmtree(work / "features")
    if not keep_training:
        state["training_stop_reason"] = decision["reason"]
        state["training_finished"] = True
        atomic_json(state_path, state)


def capture_rows(cfg, run, work, shard, state, env):
    state_path = run / "evidence/run-state.json"
    for sub in ("data", "features", "evidence"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    state.update(stage="capture", shard=shard["key"])
    atomic_json(state_path, state)
    shard_started = time.time()
    rows = runtime.capture(
        cfg,
        shard["path"],
        work / "features",
        env,
        work / "evidence/capture.log",
    )
    capture_seconds = time.time() - shard_started
    attach_monitor(run, work, rows)
    return rows, capture_seconds, shard_started


def train_rows(cfg, plan, run, work, shard, state, env):
    state_path = run / "evidence/run-state.json"
    epoch = state["epoch"]
    state["stage"] = "training"
    atomic_json(state_path, state)
    cmd = training_command(cfg, plan, run, work, shard, epoch, state["learning_rate"])
    training_env = dict(env, CUDA_VISIBLE_DEVICES=cfg["gpu_uuids"][0])
    runtime.run_stage(cmd, training_env, work / "evidence/training.log", cfg)
    metrics = json.loads((run / "outputs/training.json").read_text())
    assert metrics["stage_key"] == shard["key"]
    metric_guard(metrics["baseline"], metrics["last"])
    atomic_json(work / "evidence/committed-training.json", metrics)
    return metrics
