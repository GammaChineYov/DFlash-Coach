"""DFlash model implementation; see licenses/DFlash-MIT.txt (MIT, Z Lab)."""

import torch
from transformers.models.qwen3.modeling_qwen3 import rotate_half


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [round(start + (i * span) / (num_draft_layers - 1)) for i in range(num_draft_layers)]


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: list[int] | None,
) -> torch.Tensor:
    offset = 1
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return torch.cat(selected_states, dim=-1)


def _draft_config(config):
    return getattr(config, "dflash_config", {})


def _draft_value(config, name, default=None):
    return _draft_config(config).get(name, getattr(config, name, default))


def _attention_mask(query, key, *, is_causal, sliding_window):
    query_position = key.shape[-2] - query.shape[-2] + torch.arange(
        query.shape[-2],
        device=query.device,
    )[:, None]
    key_position = torch.arange(key.shape[-2], device=query.device)[None, :]
    visible = torch.ones(
        (query.shape[-2], key.shape[-2]),
        dtype=torch.bool,
        device=query.device,
    )
    if is_causal:
        visible &= key_position <= query_position
    if sliding_window is not None:
        visible &= query_position - key_position < sliding_window
        if not is_causal:
            visible &= key_position - query_position < sliding_window
    return visible[None, None]


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
