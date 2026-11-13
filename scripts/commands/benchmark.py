#!/usr/bin/env python3
"""Compare control, parent and candidate on reserved test prompts."""

import argparse
import json
import os
import signal
from pathlib import Path

from scripts.benchmarking.measurements import measure_variant
from scripts.benchmarking.server import BenchmarkServer
from scripts.benchmarking.server import variant_command as variant_command
from scripts.benchmarking.summary import summarize
from scripts.coach_core.configuration import command_argv, load_config
from scripts.coach_core.files import atomic_json, read_rows


def interrupted(signum, frame):
    raise KeyboardInterrupt("Benchmark interrupted")


def benchmark(cfg):
    run = Path(cfg["run_dir"])
    rows = list(read_rows(run / "data/benchmark-test.jsonl"))
    assert rows and all(row["dataset_split"] == "test" for row in rows)

    environment = dict(os.environ, **cfg.get("environment", {}))
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(cfg["gpu_uuids"])
    server = BenchmarkServer(cfg["benchmark_url"])
    command = command_argv(cfg)
    evidence_path = run / "evidence/benchmark-measurements.jsonl"
    evidence_path.write_text("")
    measurements = []
    variants = (
        ("no_draft", None),
        ("parent", cfg["initial_draft"]),
        ("candidate", run / "outputs" / cfg["output_name"]),
    )

    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, interrupted)
    try:
        for label, draft in variants:
            log_path = run / "evidence" / ("benchmark-" + label + ".log")
            with log_path.open("w") as log:
                argv = variant_command(command, draft, server.url)
                server.start(argv, environment, log)
                server.wait_ready(label)
                measured = measure_variant(server, label, rows, evidence_path)
                measurements.extend(measured)
                server.stop()
    finally:
        server.stop()

    summary = summarize(rows, measurements)
    atomic_json(run / "outputs/benchmark-summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    summary = benchmark(load_config(args.config))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
