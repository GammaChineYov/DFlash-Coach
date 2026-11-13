"""JSON/JSONL storage, private permissions and artifact hashes."""

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from scripts.coach_core.types import PathLike, Record


def digest(path: PathLike) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: PathLike, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.chmod(0o600)
    if os.geteuid() == 0:
        owner = path.parent.stat()
        os.chown(temp, owner.st_uid, owner.st_gid)
    temp.replace(path)


def read_rows(path: PathLike) -> Iterator[Record]:
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_rows(path: PathLike, rows: Iterable[Record]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    path.chmod(0o600)
    if os.geteuid() == 0:
        owner = path.parent.stat()
        os.chown(path, owner.st_uid, owner.st_gid)
