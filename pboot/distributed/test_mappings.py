import os

import pytest
import torch
import torch.distributed as dist

from pboot.distributed import (
    initialize_distributed,
    destroy_distributed,
    print_with_rank,
    all_gather,
    all_reduce,
    reduce_scatter,
    broadcast,
    send,
    recv,
    async_send_and_recv,
)


def _init_dist(device):
    mesh_def = (int(os.environ.get("WORLD_SIZE", 1)),)
    mesh_dim_names = ("global",)
    initialize_distributed(
        device,
        mesh_def,
        mesh_dim_names,
    )


def _destroy_dist():
    destroy_distributed()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_initialize_distributed(device):
    initialize_distributed(
        device,
        (2, 2, 2, 2),
        ("global", "pp", "dp", "tp"),
    )

    destroy_distributed()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_all_gather(device):
    _init_dist(device)

    print_with_rank("test_all_gather")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tensor = torch.ones(1) * rank

    res = all_gather(tensor, group=None, dim=0)
    compare = torch.arange(world_size)
    assert torch.equal(res, compare), f"{res} != {compare}"

    _destroy_dist()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_all_reduce(device):
    _init_dist(device)

    print_with_rank("test_all_reduce")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tensor = torch.ones(1) * rank

    res = all_reduce(tensor, group=None, op="avg")
    compare = torch.arange(world_size).float().mean()
    assert res.item() == compare, f"{res} != {compare}"

    _destroy_dist()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reduce_scatter(device):
    _init_dist(device)

    print_with_rank("test_reduce_scatter")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    block_size = 2

    tensor = torch.arange(world_size * block_size)

    res = reduce_scatter(tensor, group=None, dim=0)
    compare = torch.arange(rank * block_size, rank * block_size + block_size).float()
    assert torch.equal(res, compare), f"{res} != {compare}"

    _destroy_dist()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_broadcast(device):
    _init_dist(device)

    print_with_rank("test_broadcast")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        tensor = torch.arange(world_size)
    else:
        # gloo throws buffer size mistmatch error without long
        tensor = torch.empty(world_size).long()

    res = broadcast(tensor, src=0, group=None)
    compare = torch.arange(world_size)
    assert torch.equal(res, compare), f"{res} != {compare}"

    _destroy_dist()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_send_and_recv_unbatched(device):
    _init_dist(device)

    print_with_rank("test_send_and_recv_unbatched")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if 0 <= rank < world_size // 2:
        send_tensor = torch.ones(1) * rank
        dst_rank = rank + (world_size // 2)
        send(send_tensor, dst=dst_rank, group=None)

    else:
        recv_tensor = torch.empty(1)
        src_rank = rank - (world_size // 2)
        recv(recv_tensor, src=src_rank, group=None)
        assert recv_tensor.item() == src_rank

    _destroy_dist()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_send_and_recv_batched(device):
    _init_dist(device)

    print_with_rank("test_send_and_recv_batched")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    send_tensor = torch.ones(1) * rank
    recv_src = rank - 1
    if recv_src == -1:
        recv_src = world_size - 1
    send_dst = rank + 1
    if send_dst == world_size:
        send_dst = 0

    recv_tensor = async_send_and_recv(send_tensor, recv_src, send_dst, group=None)

    assert recv_tensor.item() == recv_src

    _destroy_dist()
