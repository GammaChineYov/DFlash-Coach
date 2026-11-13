"""Validate an explicit cross-plan continuation from the published best step."""
import torch


def validate_seed(saved, plan_hash, step):
    if saved['plan_hash'] != plan_hash:
        raise RuntimeError('Seed checkpoint does not belong to the recorded parent plan')
    if saved['step'] != step or saved['best_step'] != step:
        raise RuntimeError('Seed must be the selected best step, including optimizer state')
    if set(saved['model']) != set(saved['best_model']):
        raise RuntimeError('Seed model keys do not match the best model')
    for name, value in saved['model'].items():
        if not torch.isfinite(value).all() or not torch.equal(value.half(), saved['best_model'][name]):
            raise RuntimeError('Seed weights do not match the published best: ' + name)
    for state in saved['optimizer']['state'].values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise RuntimeError('Non-finite seed optimizer state')
    return saved
