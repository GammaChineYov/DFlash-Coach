"""Child processes, feature capture and restoration of the configured service."""

import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from scripts.coach_core.commands import capture_command
from scripts.coach_core.constants.recipe import (
    FEATURE_BYTES_PER_TOKEN,
    FEATURE_LAYER_COUNT,
    HIDDEN_SIZE,
)
from scripts.coach_core.files import read_rows

CHILD = None


def stop_child():
    global CHILD
    if CHILD and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)
        try:
            CHILD.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(CHILD.pid, signal.SIGKILL)
            CHILD.wait(timeout=10)
    CHILD = None


def local_api(path, data=None, port=1234, timeout=10, base_url=None):
    req = urllib.request.Request(
        (base_url or f"http://127.0.0.1:{port}").rstrip("/") + path,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def run_stage(cmd, env, log, cfg):
    global CHILD
    with Path(log).open("a") as stream:
        CHILD = subprocess.Popen(
            cmd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while CHILD.poll() is None:
            fs = os.statvfs(cfg["run_dir"])
            if fs.f_bavail <= 0:
                stop_child()
                raise RuntimeError("Filesystem is full; progress retained")
            time.sleep(2)
        rc = CHILD.returncode
        CHILD = None
    if rc:
        raise RuntimeError(f"Stage failed ({rc}); see {log}")


def verify_features(rows, directory, max_new):
    for row in rows:
        p = Path(directory) / (row["id"] + ".json")
        meta = json.loads(p.read_text())
        assert meta["prompt"] == row["prompt"]
        assert meta["source_sha256"] == row["source_sha256"]
        assert meta["feature_dtype"] == "float16"
        assert meta["max_new_tokens"] == max_new
        assert meta["feature_shape"] == [len(meta["input_ids"]), FEATURE_LAYER_COUNT, HIDDEN_SIZE]
        assert meta["prompt_n"] == row["prompt_tokens"]
        expected_bytes = len(meta["input_ids"]) * FEATURE_BYTES_PER_TOKEN
        assert p.with_suffix(".f16").stat().st_size == expected_bytes


def capture(cfg, prompts, directory, env, log):
    rows = list(read_rows(prompts))
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    # Existing metadata is reusable only after exact prompt/hash/length/size verification.
    for row in rows:
        if (directory / (row["id"] + ".json")).exists():
            verify_features([row], directory, cfg["max_new_tokens"])
    run_stage(
        capture_command(cfg, rows), dict(
            env,
            DFLASH_PROMPTS=str(prompts),
            DFLASH_FEATURES=str(directory),
        ), log, cfg
    )
    verify_features(rows, directory, cfg["max_new_tokens"])
    return rows


def service_control(action, service, scope="user"):
    if not service:
        return
    command = ["/usr/bin/systemctl"]
    if scope == "user":
        command += ["--user"]
    elif scope != "system":
        raise ValueError("production_scope must be user or system")
    command += [action, service]
    if scope == "system" and os.geteuid() != 0:
        command = ["sudo", "-n"] + command
    subprocess.run(command, check=True)


def restore(cfg, marker):
    stop_child()
    if marker.exists():
        service_control(
            "start",
            cfg["production_service"],
            cfg.get("production_scope", "user"),
        )
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                health = local_api(
                    "/health", timeout=1, base_url=cfg.get("production_url"),
                )
                if health.get("status") == "ok":
                    marker.unlink()
                    return
            except (OSError, ValueError):
                pass
            time.sleep(1)
        raise RuntimeError("target model service restart did not become healthy")
