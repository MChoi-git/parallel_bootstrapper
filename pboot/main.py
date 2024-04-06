import torch
import torch.nn as nn
import torch.distributed as dist

from pboot.config import initialize_global_config, get_config
from pboot.trainers import PretrainingTrainer
from pboot.modules import Transformer, TransformerLM
from pboot.data import (
    SyntheticMapDataset,
    CheckpointableDistributedSampler,
    tensor_collate_fn,
    build_data_iterable,
)
from pboot.optimizers import (
    build_torch_adam,
    build_constant_lr_scheduler,
)
from pboot.profiling import profile_function


def build_transformer(config):
    config.attention_mask = None
    model = TransformerLM(
        config.vocab_size,
        config.max_seq_len,
        config.nlayers,
        config.model_dim,
        config.nheads,
        config.mlp_expansion_scale,
        config.attention_dropout,
        config.mha_dropout,
        config.mlp_dropout,
        attention_mask=config.attention_mask,
        dtype=config.dtype,
        device=config.device,
    )
    print(model)
    return model


def build_dataloader(config):
    world_size = 1
    dataset = SyntheticMapDataset(
        config.vocab_size,
        config.max_seq_len,
        config.seed,
        config.num_train_samples,
    )
    sampler = CheckpointableDistributedSampler(
        len(dataset),
        rank=0,
        world_size=world_size,
        seed=config.seed,
        consumed_samples=config.consumed_samples,
    )
    dataloader = build_data_iterable(
        dataset,
        sampler,
        tensor_collate_fn,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=not config.no_pinned_memory,
        drop_last=not config.no_drop_last,
    )
    return dataloader


def build_optimizer_and_scheduler(config, model):
    optim_str_to_optim = {
        "torch_adam": build_torch_adam,
    }
    scheduler_str_to_scheduler = {
        "constant": build_constant_lr_scheduler,
    }
    build_optim = optim_str_to_optim[config.optim]
    build_scheduler = scheduler_str_to_scheduler[config.lr_scheduler]

    optim = build_optim(config, model)
    scheduler = build_scheduler(config, optim)

    return optim, scheduler


def main():
    initialize_global_config()
    config = get_config()

    trainer = PretrainingTrainer(
        config,
        build_transformer,
        build_dataloader,
        build_optimizer_and_scheduler,
    )

    trainer.train()


if __name__ == "__main__":
    main()
