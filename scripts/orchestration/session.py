"""Load run state and prepare the service, signals and child environment."""

import json
import os
import signal
import subprocess

from scripts.orchestration import runtime


def load_run_state(cfg, state_path):
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {
        "status": "running",
        "epoch": cfg.get("initial_epoch", 0),
        "next_shard": 0,
        "committed": [],
        "learning_rate": cfg["learning_rate"],
        "history": [],
    }


def install_interrupt_handlers():
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Training interrupted; restoring target model")
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, interrupted)


def ensure_production_idle(cfg):
    if cfg.get("production_service"):
        prefix = ["systemctl"]
        if cfg.get("production_scope", "user") == "user":
            prefix.append("--user")
        command = prefix + ["is-active", "--quiet", cfg["production_service"]]
        if subprocess.run(command).returncode:
            raise RuntimeError("The configured production service must initially be active")
        slots = runtime.local_api("/slots", base_url=cfg.get("production_url"))
        assert not any(slot.get("is_processing") for slot in slots)


def training_environment(cfg, plan):
    env = dict(os.environ, **cfg.get("environment", {}))
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(cfg["gpu_uuids"]),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        DFLASH_F16="1",
        DFLASH_MAX_NEW=str(cfg["max_new_tokens"]),
        DFLASH_MAX_PROMPT=str(plan["max_prompt_tokens"]),
    )
    if cfg.get("gguf_python_dir"):
        env["PYTHONPATH"] = cfg["gguf_python_dir"] + os.pathsep + env.get("PYTHONPATH", "")
    return env


def load_recovery_context(run, state, previous_error):
    recovery_context = None
    recovery_file = run / "evidence/recovery-review.json"
    if previous_error and recovery_file.exists():
        candidate_context = json.loads(recovery_file.read_text())
        if candidate_context.get("failed_shard") == state.get("shard"):
            recovery_context = candidate_context
    return recovery_context
