from argparse import Namespace

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

from pboot.distributed import (
    get_device_mesh,
    all_reduce,
)


class f(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: Tensor):
        return inputs

    @staticmethod
    def backward(ctx, inputs_grad: Tensor):
        tp_group = get_device_mesh().get_group("tp")
        out = all_reduce(inputs_grad, group=tp_group)
        return out


class g(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: Tensor):
        tp_group = get_device_mesh().get_group("tp")
        out = all_reduce(inputs, group=tp_group)
        return out

    @staticmethod
    def backward(ctx, inputs_grad: Tensor):
        return inputs_grad


class ColumnParallelLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True, dtype: torch.dtype = torch.float32, device: str = "cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dtype = dtype
        self.device = device

        self.mesh = get_device_mesh()
        tp_group = self.mesh.get_group("tp")
        tp_size = dist.get_world_size(tp_group)

        self.linear = nn.Linear(
            self.input_dim,
            self.output_dim // tp_size,
            bias=False,
            dtype=dtype,
            device=device,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        out = f.apply(inputs)
        out = self.linear(out)
        return out


class RowParallelLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True, dtype: torch.dtype = torch.float32, device: str = "cpu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dtype = dtype
        self.device = device

        self.mesh = get_device_mesh()
        tp_group = self.mesh.get_group("tp")
        tp_size = dist.get_world_size(tp_group)

        self.linear = nn.Linear(
            self.input_dim // tp_size,
            self.output_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.linear(inputs)
        out = g.apply(out)
        return out


class ParallelMLP(nn.Module):
    def __init__(self, config: Namespace):
        super().__init__()
        self.model_dim = config.model_dim
        self.hidden_dim = int(self.model_dim * config.mlp_expansion_scale)
        self.use_bias = config.use_linear_bias
        self.dtype = config.dtype
        self.device = config.device

        self.fc1 = ColumnParallelLinear(
            self.model_dim,
            self.hidden_dim,
            bias=self.use_bias,
            dtype=self.dtype,
            device=self.device,
        )
        self.fc2 = RowParallelLinear(
            self.hidden_dim,
            self.model_dim,
            bias=self.use_bias,
            dtype=self.dtype,
            device=self.device,
        )
        self.act_fn = torch.nn.functional.relu

    def forward(self, inputs):
        return self.fc2(self.act_fn(self.fc1(inputs)))
