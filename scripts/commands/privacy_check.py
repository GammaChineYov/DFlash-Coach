#!/usr/bin/env python3
"""Check the source tree for common publication mistakes without printing matches."""

import argparse
import json
from pathlib import Path

from scripts.coach_core.constants.privacy import FORBIDDEN_SUFFIXES, IGNORED_DIRECTORIES, PATTERNS


def inspect(root):
    root = Path(root)
    findings = []
    files = 0
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in IGNORED_DIRECTORIES for part in rel.parts) or not path.is_file():
            continue
        files += 1
        if path.is_symlink():
            findings.append({"file": str(rel), "kind": "symlink"})
            continue
        private_suffix = path.suffix.lower() in FORBIDDEN_SUFFIXES
        private_config = path.name == ".env" or path.name.endswith(".local.json")
        if private_suffix or private_config:
            findings.append({"file": str(rel), "kind": "private_or_binary_artifact"})
            continue
        if any(part in ("runs", "models", "data", "verification") for part in rel.parts):
            findings.append({"file": str(rel), "kind": "runtime_artifact"})
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            findings.append({"file": str(rel), "kind": "binary_file"})
            continue
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append({"file": str(rel), "kind": name})
    return {
        "status": "passed" if not findings else "failed",
        "files_checked": files,
        "findings": findings,
        "scope": "Static source check. It does not prove the absence of every identifying detail.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    report = inspect(args.root)
    print(json.dumps(report, indent=2))
    if report["findings"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
