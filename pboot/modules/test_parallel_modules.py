from argparse import Namespace
import os

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist

from pboot.distributed import (
    initialize_distributed,
    destroy_distributed,
    get_device_mesh,
)
from pboot.modules import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelMLP,
    MLP,
)


@pytest.fixture
def config():
    config = Namespace()
    config.model_dim = 64
    config.mlp_expansion_scale = 4
    config.batch_size = 4
    config.use_linear_bias = False
    return config


def _init_tp(device: str) -> None:
    torch.manual_seed(0)
    mesh_def = (int(os.environ.get("WORLD_SIZE", 1)),)
    mesh_dim_names = ("tp",)
    initialize_distributed(
        device,
        mesh_def,
        mesh_dim_names,
    )


def _get_rank(dim):
    return dist.get_rank(get_device_mesh().get_group(dim))


def _get_size(dim):
    return dist.get_world_size(get_device_mesh().get_group(dim))


def _destroy_tp():
    destroy_distributed()


def _set_weights_ones(model):
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(1.)

@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_ColumnParallelLinear(device, dtype, config):
    _init_tp(device)

    parallel_layer = ColumnParallelLinear(
        config.model_dim,
        int(config.model_dim * config.mlp_expansion_scale),
        dtype=dtype,
        device=device,
    )
    _set_weights_ones(parallel_layer)
    layer = nn.Linear(
        config.model_dim,
        int(config.model_dim * config.mlp_expansion_scale),
        bias=config.use_linear_bias,
        dtype=dtype,
        device=device,
    )
    _set_weights_ones(layer)

    inputs = torch.randn((config.batch_size, config.model_dim), dtype=dtype, device=device)

    tp_rank = _get_rank("tp")
    tp_size = _get_size("tp")

    parallel_out = parallel_layer(inputs)
    out = layer(inputs)
    local_out = out.chunk(tp_size, dim=-1)[tp_rank]

    torch.testing.assert_close(parallel_out, local_out, atol=1e-3, rtol=1e-2)

    _destroy_tp()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_RowParallelLinear(device, dtype, config):
    _init_tp(device)

    parallel_layer = RowParallelLinear(
        int(config.model_dim * config.mlp_expansion_scale),
        config.model_dim,
        dtype=dtype,
        device=device,
    )
    _set_weights_ones(parallel_layer)
    layer = nn.Linear(
        int(config.model_dim * config.mlp_expansion_scale),
        config.model_dim,
        bias=config.use_linear_bias,
        dtype=dtype,
        device=device,
    )
    _set_weights_ones(layer)

    tp_rank = _get_rank("tp")
    tp_size = _get_size("tp")

    inputs = torch.randn((config.batch_size, int(config.model_dim * config.mlp_expansion_scale)), dtype=dtype, device=device)
    local_inputs = inputs.chunk(tp_size, dim=-1)[tp_rank]

    parallel_out = parallel_layer(local_inputs)
    out = layer(inputs)

    torch.testing.assert_close(parallel_out, out, atol=1e-3, rtol=1e-2)

    _destroy_tp()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_ParallelMLP(device, dtype, config):
    _init_tp(device)

    config.dtype = dtype
    config.device = device

    parallel_layer = ParallelMLP(config)
    _set_weights_ones(parallel_layer)
    layer = MLP(config)
    _set_weights_ones(layer)

    tp_rank = _get_rank("tp")
    tp_size = _get_size("tp")

    inputs = torch.randn((config.batch_size, config.model_dim), dtype=dtype, device=device)

    parallel_out = parallel_layer(inputs)
    out = layer(inputs)

    torch.testing.assert_close(parallel_out, out, atol=1e-3, rtol=1e-2)

    parallel_out.mean().backward()
    out.mean().backward()

    parallel_grad_fc1 = parallel_layer.fc1.linear.weight.grad.clone().detach()
    parallel_grad_fc2 = parallel_layer.fc2.linear.weight.grad.clone().detach()
    grad_fc1 = layer.fc1.weight.grad.clone().detach().chunk(tp_size, dim=0)[tp_rank]
    grad_fc2 = layer.fc2.weight.grad.clone().detach().chunk(tp_size, dim=1)[tp_rank]

    torch.testing.assert_close(
        parallel_grad_fc1,
        grad_fc1,
        atol=1e-3,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        parallel_grad_fc2,
        grad_fc2,
        atol=1e-3,
        rtol=1e-2,
    )

    _destroy_tp()
