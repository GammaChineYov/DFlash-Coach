"""Own the inference child process used by one benchmark variant."""

import json
import os
import signal
import subprocess
import time
import urllib.request
from urllib.parse import urlparse

from scripts.coach_core.constants.configuration import DEFAULT_CONFIG


class BenchmarkServer:
    def __init__(self, url):
        self.url = url
        self.process = None

    def api(self, path, data=None, timeout=120):
        request = urllib.request.Request(
            self.url.rstrip("/") + path,
            data=None if data is None else json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def stop(self):
        if self.process and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        self.process = None

    def start(self, command, environment, log):
        self.process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def wait_ready(self, label):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline and self.process.poll() is None:
            try:
                if self.api("/health", timeout=1).get("status") == "ok":
                    return
            except (OSError, ValueError):
                pass
            time.sleep(1)
        raise RuntimeError("Benchmark server did not become healthy: " + label)


def variant_command(original, draft, api_url=DEFAULT_CONFIG["benchmark_url"]):
    argv = list(original)
    for option, value in (
        ("--port", str(urlparse(api_url).port or 1238)),
        ("--spec-type", "none" if draft is None else "draft-dflash"),
    ):
        argv[argv.index(option) + 1] = value
    if draft is None:
        out, i = [], 0
        while i < len(argv):
            if argv[i].startswith("--spec-draft-"):
                i += 2
            else:
                out.append(argv[i])
                i += 1
        return out
    argv[argv.index("--spec-draft-model") + 1] = str(draft)
    return argv
