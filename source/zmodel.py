"""DFlash model implementation; see licenses/DFlash-MIT.txt (MIT, Z Lab)."""

import time
from types import SimpleNamespace
from typing import ClassVar

import torch
from torch import nn
from torch.nn import functional as F
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

if __package__:
    from .layers import (
        CandidateSelector,
        GroupedDynamicCausalConv,
        Qwen3DFlashAttention,
        Qwen3DFlashDecoderLayer,
        _grouped_dynamic_convolve,
    )
    from .model_utils import (
        _attention_mask,
        _draft_config,
        _draft_value,
        apply_rotary_pos_emb,
        build_target_layer_ids,
        extract_context_feature,
    )
    from .sampling import _rejection_sample, _sample_probs, _sampling_probs, _validate_sampling, sample
else:
    from layers import (
        CandidateSelector,
        GroupedDynamicCausalConv,
        Qwen3DFlashAttention,
        Qwen3DFlashDecoderLayer,
        _grouped_dynamic_convolve,
    )
    from model_utils import (
        _attention_mask,
        _draft_config,
        _draft_value,
        apply_rotary_pos_emb,
        build_target_layer_ids,
        extract_context_feature,
    )
    from sampling import _rejection_sample, _sample_probs, _sampling_probs, _validate_sampling, sample

__all__ = [
    "CandidateSelector",
    "GroupedDynamicCausalConv",
    "Qwen3DFlashAttention",
    "Qwen3DFlashDecoderLayer",
    "_grouped_dynamic_convolve",
    "_attention_mask",
    "_draft_config",
    "_draft_value",
    "apply_rotary_pos_emb",
    "build_target_layer_ids",
    "extract_context_feature",
    "_rejection_sample",
    "_sample_probs",
    "_sampling_probs",
    "_validate_sampling",
    "sample",
    "DFlashDraftModel",
    "DFlash2DraftModel",
    "dflash_generate",
]

# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------


def _raw_input_embeddings(target: nn.Module, input_ids: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return F.embedding(input_ids, target.get_input_embeddings().weight) * scale


def _output_head(target: nn.Module) -> nn.Module:
    head = getattr(target, "lm_head", None)
    return head if head is not None else target.get_output_embeddings()


def _make_cache(config):
    cache = DynamicCache(config=config)
    cache.activate_past_recording()
    return cache


def _crop_to(cache, length):
    remove = cache.get_seq_length() - length
    cache.crop(-remove)


def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def dflash_generate(
    model: "DFlashDraftModel",
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: list[int] | None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    block_size: int | None = None,
    return_stats: bool = False,
):
    _validate_sampling(temperature, top_p, top_k)
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size if block_size is None else block_size

    output_ids = torch.full(
        (1, max_length + 1),
        model.mask_token_id,
        dtype=torch.long,
        device=target.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    past_key_values_target = _make_cache(target.config)
    past_key_values_draft = _make_cache(model.config)

    prefill_start = _cuda_time() if return_stats else None
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=block_size > 1,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
        output.logits,
        temperature,
        top_p,
        top_k,
    )
    if block_size > 1:
        target_hidden = extract_context_feature(
            output.hidden_states,
            model.target_layer_ids,
        )
    _crop_to(past_key_values_target, num_input_tokens)
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None

    decode_start = _cuda_time() if return_stats else None
    acceptance_lengths = []
    start = num_input_tokens
    stop_tokens = (
        torch.tensor(stop_token_ids, dtype=output_ids.dtype, device=output_ids.device)
        if stop_token_ids is not None
        else None
    )

    stopped = stop_tokens is not None and torch.isin(output_ids[:, start], stop_tokens).any()
    while start + 1 < max_length and not stopped:
        verify_size = min(block_size, max_length - start)
        block_output_ids = output_ids[:, start : start + verify_size].clone()
        block_position_ids = position_ids[:, start : start + verify_size]
        if verify_size > 1:
            noise_embedding = _raw_input_embeddings(
                target,
                block_output_ids,
                float(_draft_value(model.config, "input_embedding_scale", 1.0)),
            )
            draft_hidden = model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, start - target_hidden.shape[1] : start + verify_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
            )[:, 1 - verify_size :, :]
            _crop_to(past_key_values_draft, start)
            if isinstance(model, DFlash2DraftModel):
                draft_tokens, draft_indices, draft_probs = model.propose(
                    draft_hidden,
                    block_output_ids[:, 0],
                    _output_head(target),
                    temperature,
                )
                block_output_ids[:, 1:] = draft_tokens
            else:
                draft_logits = model.compute_logits(draft_hidden, _output_head(target))
                if temperature > 0:
                    draft_probs = _sampling_probs(draft_logits, temperature, top_p, top_k)
                    block_output_ids[:, 1:] = _sample_probs(draft_probs)
                    draft_indices = None
                else:
                    block_output_ids[:, 1:] = torch.argmax(draft_logits, dim=-1)
        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=verify_size > 1,
        )

        if temperature > 0:
            target_probs = _sampling_probs(output.logits, temperature, top_p, top_k)
            if verify_size > 1:
                acceptance_length, bonus = _rejection_sample(
                    block_output_ids[:, 1:],
                    target_probs,
                    draft_probs,
                    draft_indices,
                )
            else:
                acceptance_length = 0
                bonus = _sample_probs(target_probs[:, -1])[0]
        else:
            posterior = torch.argmax(output.logits, dim=-1)
            acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
            bonus = posterior[:, acceptance_length][0]
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = bonus
        produced = min(acceptance_length + 1, max_length - start - 1)
        if stop_tokens is not None:
            stop_indices = torch.isin(
                output_ids[0, start + 1 : start + produced + 1],
                stop_tokens,
            ).nonzero(
                as_tuple=True
            )[0]
            if stop_indices.numel() > 0:
                produced = stop_indices[0].item() + 1
                stopped = True
        start += produced
        _crop_to(past_key_values_target, start)
        acceptance_lengths.append(produced)

        if verify_size > 1:
            target_hidden = extract_context_feature(
                output.hidden_states,
                model.target_layer_ids,
            )[:, :produced, :]

    output_ids = output_ids[:, : min(start + 1, max_length)]

    if not return_stats:
        return output_ids

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = _cuda_time() - decode_start
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / num_output_tokens,
        acceptance_lengths=acceptance_lengths,
    )


# ---------------------------------------------------------------------------
# DFlash model
# ---------------------------------------------------------------------------


class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules: ClassVar[list[str]] = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_layer_ids = _draft_value(
            config,
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = int(_draft_value(config, "block_size", 16))
        self.mask_token_id = _draft_value(config, "mask_token_id")
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        noise_embedding: torch.Tensor | None = None,
        target_hidden: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    def compute_logits(self, hidden, output_head):
        logits = output_head(hidden)
        logits = logits * float(_draft_value(self.config, "output_multiplier", 1.0))
        softcap = _draft_value(self.config, "final_logit_softcapping")
        if softcap is not None and float(softcap) > 0:
            logits = torch.tanh(logits / float(softcap)) * float(softcap)
        return logits

    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ):
        self.eval()
        return dflash_generate(
            self,
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )


class DFlash2DraftModel(DFlashDraftModel):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault(
            "key_mapping",
            {
                f"candidate_selector.{name}": f"candidate_selector.{name}.weight"
                for name in ("predecessor_codebook", "successor_codebook")
            },
        )
        return super().from_pretrained(*args, **kwargs)

    def __init__(self, config) -> None:
        super().__init__(config)
        kernel_size = int(_draft_value(config, "conv_kernel_size"))
        group_size = int(_draft_value(config, "conv_group_size"))
        for layer in self.layers:
            layer.attention_conv = GroupedDynamicCausalConv(
                config.hidden_size,
                kernel_size,
                group_size,
            )
            layer.mlp_conv = GroupedDynamicCausalConv(
                config.hidden_size,
                kernel_size,
                group_size,
            )
        self.candidate_selector = CandidateSelector(config)
        self.post_init()

    def propose(self, hidden, anchor_ids, output_head, temperature):
        return self.candidate_selector.select(
            hidden,
            self.compute_logits(hidden, output_head),
            anchor_ids,
            temperature,
        )
