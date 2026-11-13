"""Inspect the configured training interpreter without enabling its GPU."""

import json
import os
import subprocess


def training_versions(cfg):
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        **cfg.get("environment", {}),
    )
    if cfg.get("gguf_python_dir"):
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = cfg["gguf_python_dir"] + os.pathsep + existing

    program = """
import json
import torch
import numpy
import transformers
import gguf

print(json.dumps({
    "torch": torch.__version__,
    "numpy": numpy.__version__,
    "transformers": transformers.__version__,
}))
"""
    output = subprocess.check_output(
        [cfg["python"], "-c", program],
        env=environment,
        text=True,
    )
    return json.loads(output)
