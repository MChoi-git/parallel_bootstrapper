from dataclasses import dataclass
from functools import reduce
import os

import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed.device_mesh import (
    init_device_mesh,
    DeviceMesh,
)


_DEVICE_MESH = None


def print_with_rank(msg: str, print_once=True):
    rank = dist.get_rank()

    prefix = f"Global[{rank}]: "

    if print_once is True:
        if rank == 0:
            print(prefix + msg)
    else:
        print(prefix + msg)


def set_device_mesh(ldm: DeviceMesh) -> None:
    global _DEVICE_MESH
    if _DEVICE_MESH is not None:
        print_with_rank("Overriding existing _DEVICE_MESH")

    _DEVICE_MESH = ldm


def get_device_mesh() -> DeviceMesh:
    global _DEVICE_MESH
    assert _DEVICE_MESH is not None
    return _DEVICE_MESH


def _initialize_distributed_gpu():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = os.environ.get("LOCAL_RANK", 0)
    torch.cuda.set_device(local_rank)


def _initialize_distributed_cpu():
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")


def assert_mesh_can_communicate(mesh: DeviceMesh):
    from .mappings import all_gather

    for dim in mesh.mesh_dim_names:
        dim_group = mesh.get_group(dim)
        dim_rank = dist.get_rank(dim_group)
        dim_world_size = dist.get_world_size(dim_group)

        tensor = torch.ones(1) * dim_rank

        tensor = all_gather(tensor, group=dim_group, dim=0)
        assert torch.equal(tensor, torch.arange(dim_world_size)), (
            f"Global[{dist.get_rank()}]: Mesh communication check failed!"
        )
    print_with_rank("Mesh communication check successful")


def initialize_distributed(
    device: str,
    mesh_def: tuple[int, ...],
    mesh_dim_names: tuple[str, ...],
):
    # Initialize distributed backend
    if device == "cuda":
        _initialize_distributed_cpu()
    elif device == "cpu":
        _initialize_distributed_cpu()
    else:
        raise NotImplementedError(f"Unsupported device: {device}")

    # Initialize device mesh
    device_mesh = init_device_mesh(device, mesh_def, mesh_dim_names=mesh_dim_names)
    set_device_mesh(device_mesh)
    print_with_rank(f"Initialized {get_device_mesh()}")

    # Test communication all groups in a device mesh
    assert_mesh_can_communicate(get_device_mesh())

    print_with_rank("Successfully initialized distributed")


def destroy_distributed():
    set_device_mesh(None)
