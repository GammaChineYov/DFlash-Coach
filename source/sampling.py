"""DFlash model implementation; see licenses/DFlash-MIT.txt (MIT, Z Lab)."""

import torch


def _sampling_probs(
    logits: torch.Tensor,
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    scores = logits.float() / temperature
    vocab_size = scores.shape[-1]
    if 0 < top_k < vocab_size:
        scores, indices = torch.topk(scores, top_k, dim=-1)
    else:
        indices = None

    probs = torch.softmax(scores, dim=-1)
    if top_p < 1.0:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        keep = sorted_probs.cumsum(dim=-1) - sorted_probs < top_p
        sorted_probs = sorted_probs * keep
        probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    if indices is not None:
        probs = torch.zeros_like(logits, dtype=probs.dtype).scatter(-1, indices, probs)
    return probs


def _sample_probs(probs: torch.Tensor) -> torch.Tensor:
    shape = probs.shape[:-1]
    return torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(shape)


def _validate_sampling(temperature: float, top_p: float, top_k: int) -> None:
    if temperature < 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError("temperature and top_k must be non-negative, and top_p in (0, 1].")


def sample(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    _validate_sampling(temperature, top_p, top_k)
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    return _sample_probs(_sampling_probs(logits, temperature, top_p, top_k))


def _rejection_sample(
    draft_tokens: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_indices: torch.Tensor | None = None,
) -> tuple[int, torch.Tensor]:
    gamma = draft_tokens.shape[1]
    p = target_probs[:, :gamma].gather(-1, draft_tokens[..., None])[..., 0]
    if draft_indices is None:
        q = draft_probs.gather(-1, draft_tokens[..., None])[..., 0]
    else:
        q = (draft_probs * (draft_indices == draft_tokens[..., None])).sum(-1)
    accepted = (torch.rand_like(q) * q < p).to(torch.int32).cumprod(-1).sum(-1)[0].item()
    if accepted == gamma:
        return accepted, _sample_probs(target_probs[:, -1])[0]

    residual = target_probs[0, accepted].clone()
    if draft_indices is None:
        residual.sub_(draft_probs[0, accepted])
    else:
        residual.scatter_add_(0, draft_indices[0, accepted], -draft_probs[0, accepted])
    residual.clamp_min_(0)
    total = residual.sum()
    residual = torch.where(
        total > 0,
        residual / total.clamp_min(torch.finfo(residual.dtype).tiny),
        target_probs[0, accepted],
    )
    return accepted, _sample_probs(residual[None])[0]
