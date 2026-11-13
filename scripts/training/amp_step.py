"""Retry a batch after recoverable AMP overflow without changing optimizer state."""

import math

import torch


def amp_train_step(compute, parameters, optimizer, scaler, on_overflow=None):
    parameters = list(parameters)
    retries = 0
    while True:
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = compute()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss; no optimizer update applied")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients:
            raise RuntimeError("Training produced no gradients")
        finite = bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all())
        if finite:
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            return loss, metrics, retries
        if not scaler.is_enabled():
            raise RuntimeError("Non-finite gradients without AMP scaling; no update applied")
        previous_scale = scaler.get_scale()
        # unscale_ recorded non-finite gradients; GradScaler must skip this update.
        scaler.step(optimizer)
        scaler.update()
        current_scale = scaler.get_scale()
        optimizer.zero_grad(set_to_none=True)
        if not math.isfinite(current_scale) or not 0 < current_scale < previous_scale:
            raise RuntimeError("AMP could not reduce loss scale; batch not committed")
        retries += 1
        if on_overflow:
            on_overflow(retries, previous_scale, current_scale)
