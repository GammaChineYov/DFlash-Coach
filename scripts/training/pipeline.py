"""Coordinate sample validation, resume, training and the atomic shard commit."""

import json
import math
import os
import random
import time
from functools import partial

os.environ.setdefault("OMP_NUM_THREADS", "2")

import torch

from scripts.coach_core.constants.recipe import TRAINING_SEED
from scripts.coach_core.metrics import metric_guard
from scripts.training.backend import create_backend
from scripts.training.checkpoints import (
    commit_checkpoint,
    initial_progress,
    restore_checkpoint,
    select_best,
    snapshot_model,
)
from scripts.training.data import (
    load_samples,
    split_training_samples,
    validate_sample_data,
)
from scripts.training.engine import train_batches
from scripts.training.model_io import export_draft
from scripts.training.reporting import emit_progress, training_report


def restore_committed_report(args, saved, emit):
    if saved is None or args.eval_only:
        return False
    if saved["stage_key"] != args.stage_key:
        return False

    report_path = args.state_dir / "training.json"
    report_path.write_text(json.dumps(saved["result"], indent=2))
    emit("ALREADY_COMMITTED", stage_key=args.stage_key, step=saved["step"])
    return True


def evaluate_before(backend, train, valid, emit):
    metrics = backend.objective.evaluate()
    if any(not math.isfinite(float(value)) for value in metrics.values()):
        raise RuntimeError("Non-finite initial validation")

    emit(
        "VALIDATION_BEFORE",
        metrics=metrics,
        train_examples=len(train),
        eval_examples=len(valid),
    )
    return metrics


def finish_shard(args, backend, progress, batches, before, valid, started, emit):
    last = backend.objective.evaluate()
    emit("VALIDATION", step=batches.step, metrics=last)
    metric_guard(progress.baseline, last)
    select_best(progress, backend.model, last, batches.step)

    # Write the candidate and report before committing the resumable checkpoint.
    current_state = snapshot_model(backend.model)
    backend.model.load_state_dict(progress.best_model)
    export_draft(
        backend.model,
        backend.reader,
        args.state_dir / args.output_name,
        args.model_name,
    )
    result = training_report(
        args, backend, progress, batches,
        before, last, len(valid), started,
    )
    report_path = args.state_dir / "training.json"
    report_path.write_text(json.dumps(result, indent=2))
    commit_checkpoint(args, backend, progress, current_state, result)

    summary = dict(result)
    summary.pop("history")
    summary.pop("accepted_training_ids")
    emit("TRAIN_DONE", **summary)


def train(args):
    args.state_dir.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    emit = partial(emit_progress, args.run_dir)

    samples = load_samples(args.run_dir)
    training, validation = split_training_samples(args, samples)
    if args.validate_data_only:
        validate_sample_data(samples, training, validation, emit)
        return

    backend = create_backend(args, validation)
    started = time.monotonic()
    saved = restore_checkpoint(args, backend)
    if restore_committed_report(args, saved, emit):
        return
    if saved is not None:
        emit("RESUMED", step=saved["step"] + 1)

    before = evaluate_before(backend, training, validation, emit)
    if args.eval_only:
        emit("EVALUATED", metrics=before)
        return

    progress = initial_progress(saved, before, backend.model)
    del saved
    batches = train_batches(
        args, backend, training, progress.first_step, emit, started,
    )
    finish_shard(
        args, backend, progress, batches, before, validation, started, emit,
    )
