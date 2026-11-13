"""GGUF tensor mapping and chunked dequantization for the supported backend."""

import numpy as np
import torch
from scripts.coach_core.constants.gguf import DIRECT_TENSOR_NAMES, LAYER_TENSOR_NAMES
from scripts.coach_core.constants.recipe import TARGET_LAYERS as LAYERS
from gguf import dequantize

__all__ = ["LAYERS", "tensor_name", "read_rows"]


def tensor_name(name):
    if name in DIRECT_TENSOR_NAMES:
        return DIRECT_TENSOR_NAMES[name]
    parts = name.split(".")
    assert parts[0] == "layers", name
    tail = ".".join(parts[2:])
    return f"blk.{parts[1]}." + LAYER_TENSOR_NAMES[tail]


def read_rows(tensor, device):
    rows = int(tensor.shape[1])
    cols = int(tensor.shape[0])
    out = torch.empty((rows, cols), dtype=torch.float16, device=device)
    for begin in range(0, rows, 1024):
        part = dequantize(tensor.data[begin : begin + 1024], tensor.tensor_type)
        out[begin : begin + len(part)].copy_(
            torch.from_numpy(np.array(part, copy=True)).to(
                device=device,
                dtype=torch.float16,
            )
        )
    return out
