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
    ParallelMHA,
    MHA,
)


@pytest.fixture
def config():
    config = Namespace()
    config.model_dim = 8
    config.mlp_expansion_scale = 4
    config.batch_size = 4
    config.use_linear_bias = False
    config.seq_len = 6
    config.nheads = 4
    config.attention_dropout = 0.
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


def _sync_weights(model_1, model_2):
    def _get_sharded_dim(t1, t2):
        sharded = []
        for i, (d1, d2) in enumerate(zip(t1.shape, t2.shape)):
            if d1 != d2:
                sharded.append(i)
        assert len(sharded) in [0, 1]
        return sharded[0] if len(sharded) == 1 else -1

    with torch.no_grad():
        for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
            shard_dim = _get_sharded_dim(p1, p2)

            if shard_dim == -1:
                chunk = p2.clone().detach()

            else:
                tp_size = _get_size("tp")
                tp_rank = _get_rank("tp")
                chunk = p2.clone().detach().chunk(tp_size, dim=shard_dim)[tp_rank]

            p1.copy_(chunk)


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
    layer = nn.Linear(
        config.model_dim,
        int(config.model_dim * config.mlp_expansion_scale),
        bias=config.use_linear_bias,
        dtype=dtype,
        device=device,
    )
    _sync_weights(parallel_layer, layer)

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
    layer = nn.Linear(
        int(config.model_dim * config.mlp_expansion_scale),
        config.model_dim,
        bias=config.use_linear_bias,
        dtype=dtype,
        device=device,
    )
    _sync_weights(parallel_layer, layer)

    tp_rank = _get_rank("tp")
    tp_size = _get_size("tp")

    inputs = torch.randn((config.batch_size, int(config.model_dim * config.mlp_expansion_scale)), dtype=dtype, device=device)
    local_inputs = inputs.chunk(tp_size, dim=-1)[tp_rank]

    parallel_out = parallel_layer(local_inputs)
    out = layer(inputs)

    if dtype == torch.bfloat16:
        torch.testing.assert_close(parallel_out, out, atol=5e-3, rtol=6e-2)
    else:
        torch.testing.assert_close(parallel_out, out, atol=1e-3, rtol=1e-2)

    _destroy_tp()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_ParallelMLP(device, dtype, config):
    _init_tp(device)

    config.dtype = dtype
    config.device = device

    parallel_layer = ParallelMLP(config)
    layer = MLP(config)
    _sync_weights(parallel_layer, layer)

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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_ParallelMHA(device, dtype, config):
    _init_tp(device)

    config.dtype = dtype
    config.device = device

    parallel_layer = ParallelMHA(config)
    layer = MHA(config)
    _sync_weights(parallel_layer, layer)

    tp_rank = _get_rank("tp")
    tp_size = _get_size("tp")

    # Sanity-check weight sync
    assert torch.equal(layer.q_proj.weight.chunk(tp_size, dim=0)[tp_rank], parallel_layer.q_proj.linear.weight)
    assert torch.equal(layer.k_proj.weight.chunk(tp_size, dim=0)[tp_rank], parallel_layer.k_proj.linear.weight)
    assert torch.equal(layer.v_proj.weight.chunk(tp_size, dim=0)[tp_rank], parallel_layer.v_proj.linear.weight)
    assert torch.equal(layer.out_proj.weight.chunk(tp_size, dim=1)[tp_rank], parallel_layer.out_proj.linear.weight)

    inputs = torch.randn((config.batch_size, config.seq_len, config.model_dim), dtype=dtype, device=device)
    cos = torch.randn((config.batch_size, config.seq_len, config.nheads, config.model_dim // config.nheads), device=device, dtype=dtype)
    sin = torch.randn((config.batch_size, config.seq_len, config.nheads, config.model_dim // config.nheads), device=device, dtype=dtype)

    # Test forward result
    parallel_out = parallel_layer(inputs, cos, sin)
    out = layer(inputs, cos, sin)

    torch.testing.assert_close(parallel_out, out, atol=1e-3, rtol=1e-2)

    # Test backward grads
    parallel_out.mean().backward()
    out.mean().backward()

    parallel_grad_q = parallel_layer.q_proj.linear.weight.grad.clone().detach()
    parallel_grad_k = parallel_layer.k_proj.linear.weight.grad.clone().detach()
    parallel_grad_v = parallel_layer.v_proj.linear.weight.grad.clone().detach()
    parallel_grad_out = parallel_layer.out_proj.linear.weight.grad.clone().detach()

    grad_q = layer.q_proj.weight.grad.clone().detach().chunk(tp_size, dim=0)[tp_rank]
    grad_k = layer.k_proj.weight.grad.clone().detach().chunk(tp_size, dim=0)[tp_rank]
    grad_v = layer.v_proj.weight.grad.clone().detach().chunk(tp_size, dim=0)[tp_rank]
    grad_out = layer.out_proj.weight.grad.clone().detach().chunk(tp_size, dim=1)[tp_rank]

    torch.testing.assert_close(
        parallel_grad_q,
        grad_q,
        atol=1e-3,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        parallel_grad_k,
        grad_k,
        atol=1e-3,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        parallel_grad_v,
        grad_v,
        atol=1e-3,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        parallel_grad_out,
        grad_out,
        atol=1e-3,
        rtol=1e-2,
    )
    _destroy_tp()
