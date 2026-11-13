"""Read mentor credentials from the configured environment or private file."""

import os
import stat
from pathlib import Path


def read_key(cfg):
    variable = cfg.get("key_env", "DEEPSEEK_API_KEY")
    environment_value = os.environ.get(variable, "").strip()
    if environment_value:
        return environment_value
    if not cfg.get("key_file"):
        raise RuntimeError(
            "Set the configured mentor API key environment variable "
            "or a private key file"
        )
    path = Path(cfg["key_file"])
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise RuntimeError("DeepSeek credential file must be mode 0600")
    for line in path.read_text().splitlines():
        if not line.startswith(variable + "="):
            continue
        value = line.split("=", 1)[1].strip()
        if value:
            return value
    raise RuntimeError("DeepSeek credential missing")
