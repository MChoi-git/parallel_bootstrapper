from argparse import Namespace
import pytest
import torch
import torch.nn as nn


from pboot.trainers import (
    PretrainingTrainer,
)


@pytest.fixture
def config():
    config = Namespace()
    config.checkpoint_dir = "."
    config.checkpoint_prefix = "pytest"
    config.checkpoint_format = "pt"
    config.consumed_samples = 0
    config.train_steps = 10
    config.loss_log_interval = 5
    config.train_steps_per_val = 5
    config.train_steps_per_checkpoint = 11
    return config


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(2, 2, device=config.device, dtype=config.dtype)
    def forward(self, inputs):
        return self.fc(inputs).mean()


def build_model_fn(config):
    return Model(config)


def _set_weights_ones(model):
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.)


def build_dataloader_fn(config):
    fake_dset = torch.utils.data.TensorDataset(torch.randn(100, 2))
    dataloader = torch.utils.data.DataLoader(
        fake_dset,
        batch_size=4,
    )
    return dataloader


def build_optimizer_and_scheduler_fn(config, model):
    optim = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ConstantLR(optim, factor=1.)
    return optim, scheduler


@pytest.mark.parametrize("torch_compile", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_PretraingTrainer_train_step(config, torch_compile, device, dtype):
    config.device = device
    config.dtype = dtype

    trainer = PretrainingTrainer(
        config,
        build_model_fn,
        build_dataloader_fn,
        build_optimizer_and_scheduler_fn,
    )
    batch = torch.randn(4, 2, device=config.device, dtype=config.dtype)

    step_fn = torch.compile(trainer.train_step) if torch_compile is True else trainer.train_step

    loss = step_fn(batch)

    assert torch.isfinite(loss)
    assert trainer.model.fc.weight.grad is not None
    assert trainer.optim.state[trainer.model.fc.weight] is not None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_PretraingTrainer_train_step_compile(config, device, dtype):
    config.device = device
    config.dtype = dtype

    batch = torch.randn(4, 2, device=config.device, dtype=config.dtype)

    trainer = PretrainingTrainer(
        config,
        build_model_fn,
        build_dataloader_fn,
        build_optimizer_and_scheduler_fn,
    )
    _set_weights_ones(trainer.model)
    normal_loss = trainer.train_step(batch)

    _set_weights_ones(trainer.model)
    compiled_step = torch.compile(trainer.train_step)
    compiled_loss = compiled_step(batch)

    torch.testing.assert_close(normal_loss, compiled_loss)
