"""Load the fixed draft recipe and export its best weights to GGUF."""

from copy import deepcopy

import numpy as np
import torch
from scripts.coach_core.constants.recipe import DRAFT_MODEL_CONFIG
from gguf import GGUFReader, GGUFValueType, GGUFWriter
from scripts.training.gguf import read_rows, tensor_name
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from source.zmodel import DFlash2DraftModel


def load_draft(source, device):
    reader = GGUFReader(str(source))
    tensors = {t.name: t for t in reader.tensors}
    config = Qwen3Config(**deepcopy(DRAFT_MODEL_CONFIG))
    with torch.device("meta"):
        model = DFlash2DraftModel(config)
    state = {}
    for name, param in model.state_dict().items():
        value = torch.from_numpy(np.array(tensors[tensor_name(name)].data, copy=True)).to(
            device=device, dtype=torch.float32
        )
        assert value.shape == param.shape, (name, value.shape, param.shape)
        state[name] = value
    model.load_state_dict(state, assign=True)
    del state
    model.rotary_emb = Qwen3RotaryEmbedding(config, device=device)
    model.requires_grad_(True)
    return model, reader


def load_target_tables(path, device):
    target = GGUFReader(str(path))
    tt = {t.name: t for t in target.tensors}
    embedding = read_rows(tt["token_embd.weight"], device)
    head = read_rows(tt["output.weight"], device)
    return embedding, head


def export_draft(model, reader, path, model_name):
    by_name = {tensor_name(name): value for name, value in model.state_dict().items()}
    writer = GGUFWriter(str(path), "dflash")
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        value = field.contents()
        if key == "general.name":
            value = model_name
        if field.types[0] == GGUFValueType.ARRAY:
            writer.add_array(key, value)
        else:
            writer.add_key_value(key, value, field.types[0])
    for tensor in reader.tensors:
        value = by_name[tensor.name].detach().cpu().numpy().astype(tensor.data.dtype)
        writer.add_tensor(tensor.name, value, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
