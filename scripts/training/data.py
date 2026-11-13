"""Load verified sample metadata and copy bounded slices from feature files."""

import json
import os

import numpy as np
from scripts.coach_core.constants.recipe import FEATURE_LAYER_COUNT, FEATURE_WIDTH, HIDDEN_SIZE


def load_samples(run):
    input_path = run / "data/all.jsonl"
    records = map(json.loads, input_path.read_text().splitlines())
    expected = {row["id"]: row for row in records}
    samples = []
    for file in sorted((run / "features").glob("*.json")):
        sample = json.loads(file.read_text())
        reference = expected[sample["id"]]
        assert sample["prompt"] == reference["prompt"]
        assert sample["source_sha256"] == reference["source_sha256"]
        shape = sample["feature_shape"]
        assert shape[0] == len(sample["input_ids"])
        assert shape[1:] == [FEATURE_LAYER_COUNT, HIDDEN_SIZE]
        suffix = ".f16" if sample["feature_dtype"] == "float16" else ".f32"
        dtype = np.float16 if suffix == ".f16" else np.float32
        feature = file.with_suffix(suffix)
        assert feature.stat().st_size == shape[0] * FEATURE_WIDTH * np.dtype(dtype).itemsize
        if shape[0] - sample["prompt_n"] < 9:
            continue
        sample["feature_path"] = feature
        sample["feature_numpy_dtype"] = dtype
        samples.append(sample)
    return samples


def copy_feature_rows(sample, begin, end):
    # Keep descriptors bounded even when the dataset has thousands of feature files.
    mapped = np.memmap(
        sample["feature_path"],
        dtype=sample["feature_numpy_dtype"],
        mode="r",
        shape=(sample["feature_shape"][0], FEATURE_WIDTH),
    )
    try:
        return np.array(mapped[begin:end], copy=True)
    finally:
        mapped._mmap.close()


def split_training_samples(args, samples):
    training = [sample for sample in samples if sample["split"] == "train"]
    validation = [sample for sample in samples if sample["split"] == "eval"]
    enough_total = len(samples) >= args.min_total
    enough_training = len(training) >= args.min_train
    enough_validation = len(validation) >= args.min_eval
    assert enough_total and enough_training and enough_validation, (
        len(training), len(validation),
    )
    assert training and validation

    train_sources = {
        (sample["project"], sample["source_path"])
        for sample in training
    }
    validation_sources = {
        (sample["project"], sample["source_path"])
        for sample in validation
    }
    assert train_sources.isdisjoint(validation_sources)
    return training, validation


def validate_sample_data(samples, train, valid, emit):
    before_fds = len(os.listdir("/proc/self/fd"))
    peak = before_fds
    for sample in samples:
        value = copy_feature_rows(sample, 0, min(2, sample["feature_shape"][0]))
        assert np.isfinite(value).all(), sample["id"]
        peak = max(peak, len(os.listdir("/proc/self/fd")))
    after_fds = len(os.listdir("/proc/self/fd"))
    assert after_fds <= before_fds + 1, (before_fds, after_fds)
    emit(
        "DATA_VALIDATED",
        samples=len(samples),
        train_examples=len(train),
        eval_examples=len(valid),
        before_fds=before_fds,
        after_fds=after_fds,
        peak_sampled_fds=peak,
    )
