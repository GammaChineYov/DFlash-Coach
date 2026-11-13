#!/usr/bin/env python3
"""Verify exact rendered prompt lengths using the target GGUF vocabulary on CPU."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from scripts.coach_core.configuration import load_config
from scripts.coach_core.files import atomic_json, read_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run = Path(cfg["run_dir"])
    plan = json.loads((run / "plan.json").read_text())
    expected = {r["id"]: r["prompt_tokens"] for r in read_rows(run / "data/tokenize-all.jsonl")}
    env = dict(os.environ, **cfg.get("environment", {}))
    env["CUDA_VISIBLE_DEVICES"] = ""
    path = run / "evidence/token-counts.jsonl"
    with path.open("w") as out:
        subprocess.run(
            [
                cfg["tokenize_binary"],
                cfg["target_model"],
                str(run / "data/tokenize-all.jsonl"),
                str(plan["max_prompt_tokens"]),
            ],
            stdout=out,
            env=env,
            check=True,
        )
    seen = set()
    mismatches = 0
    summary = None
    for row in read_rows(path):
        if row.get("summary"):
            summary = row
        else:
            if row["id"] in seen:
                raise ValueError("Duplicate tokenizer output")
            seen.add(row["id"])
            mismatches += expected[row["id"]] != row["prompt_tokens"]
    if seen != set(expected) or summary is None or summary["invalid"] or mismatches:
        raise ValueError("Exact prompt token counts do not match; fix input prompt_tokens")
    report = {**summary, "mismatches": mismatches, "cpu_only": True}
    atomic_json(run / "evidence/token-verification.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
