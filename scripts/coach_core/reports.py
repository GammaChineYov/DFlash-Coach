"""Numerical evidence sent to the mentor at each training boundary."""

from scripts.coach_core.metrics import trend_summary
from scripts.coach_core.types import Config, Record


def startup_report(
    cfg: Config,
    plan: Record,
    state: Record,
    resume_metrics: Record | None,
    previous_error: object,
    recovery_context: Record | None,
) -> Record:
    phase = "resume_training" if state["committed"] else "before_training"
    return {
        "phase": phase,
        "cpu_preflight": "passed",
        "shards_per_epoch": len(plan["shards"]),
        "steps_per_epoch_estimate": plan["steps_per_epoch_estimate"],
        "epoch_limit": cfg.get("max_epochs"),
        "learning_rate": state["learning_rate"],
        "validation_monitor_rows": cfg["validation_monitor_rows"],
        "recent_history": state["history"][-32:],
        "continuation": cfg.get("continuation"),
        "seed_checkpoint": cfg.get("seed_step"),
        "previous_stage_error_present": bool(previous_error),
        "technical_recovery_report_available": recovery_context is not None,
        "validation_trend": trend_summary(state["history"], resume_metrics["best_step"]) if resume_metrics else None,
    }


def shard_report(
    cfg: Config,
    plan: Record,
    state: Record,
    shard: Record,
    index: int,
    metrics: Record,
    rows: list[Record],
    point: Record,
    epoch: int,
) -> Record:
    return {
        "phase": "after_shard",
        "continuation": cfg.get("continuation"),
        "shard": shard["key"],
        "epoch": epoch + 1,
        "completed_shards_in_epoch": index + 1,
        "shards_per_epoch": len(plan["shards"]),
        "epoch_limit": cfg.get("max_epochs"),
        "steps": metrics["steps"],
        "learning_rate": metrics["learning_rate"],
        "proposed_next_learning_rate": state["learning_rate"],
        "baseline": metrics["baseline"],
        "current": metrics["last"],
        "best": metrics["best"],
        "recent_history": state["history"][-32:],
        "validation_trend": trend_summary(state["history"] + [point], metrics["best_step"]),
        "train_examples": metrics["train_examples"],
        "short_generation_exclusions": len(rows) - metrics["train_examples"],
        "gpu_peak_gib": metrics["peak_memory_gib"],
        "amp_overflow_retries": metrics.get("amp_overflow_retries", 0),
        "amp_final_scale": metrics.get("amp_final_scale"),
        "errors": [],
    }


def epoch_report(cfg: Config, plan: Record, state: Record, metrics: Record, epoch: int) -> Record:
    return {
        "phase": "after_epoch",
        "continuation": cfg.get("continuation"),
        "epochs_completed": epoch + 1,
        "epoch_limit": cfg.get("max_epochs"),
        "steps_per_epoch_estimate": plan["steps_per_epoch_estimate"],
        "learning_rate": state["learning_rate"],
        "baseline": metrics["baseline"],
        "current": metrics["last"],
        "best": metrics["best"],
        "recent_history": state["history"][-32:],
        "validation_trend": trend_summary(state["history"], metrics["best_step"]),
        "instruction": "Decide whether another full pass over the configured train split is useful, within configured limits.",
    }
