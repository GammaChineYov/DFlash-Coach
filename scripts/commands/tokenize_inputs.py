#!/usr/bin/env python3
"""Populate exact token counts in normalized inputs before freezing the plan."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from scripts.coach_core.configuration import load_config
from scripts.coach_core.files import read_rows, write_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    source = Path(cfg["dataset"])
    if (Path(cfg["run_dir"]) / "plan.json").exists():
        raise SystemExit("Tokenize before preparing the immutable run")
    env = dict(os.environ, **cfg.get("environment", {}))
    env["CUDA_VISIBLE_DEVICES"] = ""
    results = {}
    maximum = cfg["capture_context"] - cfg["max_new_tokens"] - 1
    # Verify every split first; do not partially rewrite inputs on tokenizer failure.
    for split in ("train", "validation", "test"):
        rows = list(read_rows(source / (split + ".jsonl")))
        with tempfile.TemporaryFile(mode="w+") as stream:
            subprocess.run(
                [
                    cfg["tokenize_binary"],
                    cfg["target_model"],
                    str(source / (split + ".jsonl")),
                    str(maximum),
                ],
                stdout=stream,
                env=env,
                check=True,
            )
            stream.seek(0)
            counts = {}
            summary = None
            for line in stream:
                value = json.loads(line)
                if value.get("summary"):
                    summary = value
                else:
                    if value["id"] in counts:
                        raise ValueError("Duplicate tokenizer output")
                    counts[value["id"]] = value["prompt_tokens"]
        if summary is None or summary["invalid"] or set(counts) != {r["id"] for r in rows}:
            raise ValueError("Tokenizer rejected a split; check context size and duplicate IDs")
        results[split] = [
            {
                **r,
                "prompt_tokens": counts[r["id"]],
                "length_bucket": "compact" if counts[r["id"]] <= 1536 else "extended",
            }
            for r in rows
        ]
    for split, rows in results.items():
        write_rows(source / (split + ".jsonl"), rows)
    print(json.dumps({"status": "exact_token_counts_updated", "rows": sum(map(
        len,
        results.values(),
    ))}))


if __name__ == "__main__":
    main()
