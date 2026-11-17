"""Construct the CUDA resources used throughout one shard invocation."""

import torch

from scripts.coach_core.types.training import TrainingBackend
from scripts.training.engine import create_optimizer
from scripts.training.model_io import load_draft, load_target_tables
from scripts.training.objective import ShardObjective


def create_backend(args, validation):
    device = torch.device("cuda:0")
    model, reader = load_draft(args.initial_draft, device)
    # 21.4 GB 单卡：f32 参数 + 梯度 + 词表两表装不下全部可训练，冻结到只剩 fc 与最后两层
    train_prefixes = ("fc.", "layers.3.", "layers.4.")
    frozen = 0
    for name, param in model.named_parameters():
        if not name.startswith(train_prefixes):
            param.requires_grad_(False)
            frozen += 1
    print(f"[mem] frozen {frozen} params, trainable {train_prefixes}", flush=True)
    embedding, head = load_target_tables(args.target_model, device)
    objective = ShardObjective(
        model, embedding, head,
        args.context, args.batch_size, validation,
    )
    optimizer = create_optimizer(model, args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    return TrainingBackend(model, reader, objective, optimizer, scaler)
