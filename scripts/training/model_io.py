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
    import os
    pt_path = os.environ.get("DRAFT_TENSORS_PT")
    if pt_path:
        blob = torch.load(pt_path, map_location="cpu", weights_only=False)
        raw = blob.get("tensors", blob)
        raw_fields = blob.get("fields", {})

        class _F:
            __slots__ = ("types", "_value")
            def __init__(self, vtype, value):
                self.types = [vtype]; self._value = value
            def contents(self):
                return self._value

        class _T:
            __slots__ = ("name", "data", "tensor_type")
            def __init__(self, name, data, ttype):
                self.name = name; self.data = data.numpy(); self.tensor_type = ttype

        class _R:
            def __init__(self, items, fields):
                self.tensors = [_T(k, v, items.get(k + "__type__", 0))
                                for k, v in items.items() if not k.endswith("__type__")]
                self.fields = {k: _F(vt, val) for k, (vt, val) in fields.items()}

        reader = _R(raw, raw_fields)
        tensors = {t.name: t for t in reader.tensors}
    else:
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
    try:
        model.rotary_emb = Qwen3RotaryEmbedding(config, device=device)
    except TypeError:  # transformers 5.x 去掉了 device 关键字
        model.rotary_emb = Qwen3RotaryEmbedding(config).to(device)
    model.requires_grad_(True)
    return model, reader


def load_target_tables(path, device):
    import os
    st = os.environ.get("TARGET_SAFETENSORS")
    if st:
        # 恢复出的 NVFP4 权重：embed 是 BF16，lm_head 是 F8_E4M3（需按 128x128 分块反量化）
        from safetensors import safe_open
        emb_key = os.environ.get("TARGET_EMB_KEY", "model.language_model.embed_tokens.weight")
        head_key = os.environ.get("TARGET_HEAD_KEY", "lm_head.weight")
        with safe_open(st, framework="pt") as f:
            embedding = f.get_tensor(emb_key).to(device=device, dtype=torch.float16)
            head = f.get_tensor(head_key)
            if str(head.dtype).endswith("float8_e4m3fn"):
                sc_key = head_key + "_scale_inv"
                try:
                    sc = f.get_tensor(sc_key)
                except Exception:
                    sc = f.get_tensor(head_key + "_scale")  # NVFP4 检查点用的是 _scale
                sc = sc.to(torch.float32)
                hf32 = head.to(torch.float32)
                # 实测 scale 形状：[vocab, 1] 逐行缩放；若遇到 [vocab, cols/128] 分块缩放也能处理
                if sc.dim() == 2 and sc.shape[1] == 1:
                    head = (hf32 * sc).to(device=device, dtype=torch.float16)
                else:
                    s_ = sc.repeat_interleave(128, 1)[:, : hf32.shape[1]]
                    head = (hf32 * s_).to(device=device, dtype=torch.float16)
            else:
                head = head.to(device=device, dtype=torch.float16)
        return embedding, head
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
