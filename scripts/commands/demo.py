#!/usr/bin/env python3
"""Offline synthetic demonstration: no GPU, model files, network or API key."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.coach_core.files import atomic_json, write_rows
from scripts.coach_core.planning import shard_rows
from scripts.mentoring.review import apply_decision
from scripts.mentoring.strategy import trend_summary


def synthetic_rows():
    return [
        {
            "id": f"synthetic_{i:03d}",
            "prompt_tokens": 80 + i,
            "project": "example",
            "source_path": f"file_{i}.txt",
            "source_sha256": hashlib.sha256(f"synthetic:{i}".encode()).hexdigest(),
            "split": "train",
            "prompt": f"SYNTHETIC DEMO INPUT {i}. No source data.",
        }
        for i in range(24)
    ]


def simulate_shards(run, shards, state):
    for index, shard in enumerate(shards):
        key = f"demo-s{index:04d}"
        write_rows(run / "data" / (key + ".jsonl"), shard)
        point = {
            "steps": (index + 1) * 2,
            "learning_rate": state["learning_rate"],
            "metrics": {"loss": [2.2, 2.1, 2.1][index], "prefix_tokens": 1.5},
        }
        state["history"].append(point)
        state["committed"].append(key)
        decision = {
            "action": "stop" if index == 2 else "continue",
            "reason": "Synthetic offline decision; no remote model was called.",
            "learning_rate": 1e-6 if index == 1 else None,
        }
        apply_decision(state, decision)
        atomic_json(
            run / "evidence/mentor" / (key + ".json"),
            {"simulation": True, "decision": decision},
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/demo"))
    args = parser.parse_args()
    if args.run_dir.exists():
        raise SystemExit("Choose a fresh demo --run-dir")
    rows = synthetic_rows()
    shards = list(shard_rows(rows, 8, 64 * 2**20, 32))
    state = {"simulation": True, "learning_rate": 2.5e-6, "history": [], "committed": []}
    simulate_shards(args.run_dir, shards, state)
    state.update(
        status="simulation_completed",
        training_performed=False,
        network_requests=0,
        best_step=4,
        validation_trend=trend_summary(state["history"], 4),
    )
    atomic_json(args.run_dir / "evidence/run-state.json", state)
    print(
        json.dumps(
            {
                "status": state["status"],
                "samples": len(rows),
                "shards": len(shards),
                "training_performed": False,
                "network_requests": 0,
                "run_dir": str(args.run_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
