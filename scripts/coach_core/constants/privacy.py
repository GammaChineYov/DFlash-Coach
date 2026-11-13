"""Shared privacy constants."""

import re

IGNORED_DIRECTORIES = {".git", ".venv", "__pycache__"}

PATTERNS = {
    "private_key": re.compile(r"-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "access_token": re.compile(r"\b(?:hf_[A-Za-z0-9]{24,}|sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{24,})\b"),
    "personal_home": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+"),
    "mounted_volume": re.compile(r"/Volumes/[A-Za-z0-9_.-]+"),
    "private_network": re.compile(
        r"\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
    ),
    "device_uuid": re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F-]{20,}"),
}

FORBIDDEN_SUFFIXES = {
    ".gguf",
    ".safetensors",
    ".pt",
    ".pth",
    ".f16",
    ".f32",
    ".pem",
    ".key",
    ".sqlite",
    ".db",
    ".zip",
    ".tar",
    ".gz",
}
