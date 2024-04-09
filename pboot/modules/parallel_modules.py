from argparse import Namespace
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

from pboot.distributed import (
    get_device_mesh,
    all_reduce,
)
from pboot.modules import apply_rotary_position_embeddings


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


# NOTE: Column and Row parallel linear layers don't handle biases yet
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


class ParallelMHA(nn.Module):
    def __init__(self, config: Namespace):
        super().__init__()
        self.model_dim = config.model_dim
        self.nheads = config.nheads
        self.attention_dropout = config.attention_dropout
        self.use_bias = config.use_linear_bias
        self.dtype = config.dtype
        self.device = config.device

        self.apply_rope_fn = apply_rotary_position_embeddings

        self.mesh = get_device_mesh()
        tp_group = self.mesh.get_group("tp")
        self.tp_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)

        # Parallel along head dimension
        self.local_heads = self.nheads // self.tp_size
        self.head_dim = self.model_dim // self.nheads

        self.q_proj = ColumnParallelLinear(
            self.model_dim,
            self.model_dim,
            bias=self.use_bias,
            dtype=self.dtype,
            device=self.device,
        )
        self.k_proj = ColumnParallelLinear(
            self.model_dim,
            self.model_dim,
            bias=self.use_bias,
            dtype=self.dtype,
            device=self.device,
        )
        self.v_proj = ColumnParallelLinear(
            self.model_dim,
            self.model_dim,
            bias=self.use_bias,
            dtype=self.dtype,
            device=self.device,
        )

        self.out_proj = RowParallelLinear(
            self.model_dim,
            self.model_dim,
            bias=self.use_bias,
            dtype=self.dtype,
            device=self.device,
        )

    def forward(self, inputs: Tensor, cos: Tensor, sin: Tensor, attention_mask: Optional[Tensor] = None):
        B, S, H = inputs.shape

        # bsh -> bsh -> bskd -> bksd
        q = self.q_proj(inputs).reshape(B, S, self.local_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(inputs).reshape(B, S, self.local_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(inputs).reshape(B, S, self.local_heads, self.head_dim).permute(0, 2, 1, 3)

        # Chunk along head dimension similar to q and k
        cos_chunk = torch.chunk(cos, self.tp_size, dim=2)[self.tp_rank].permute(0, 2, 1, 3)
        sin_chunk = torch.chunk(sin, self.tp_size, dim=2)[self.tp_rank].permute(0, 2, 1, 3)

        q, k = self.apply_rope_fn(q, k, cos_chunk, sin_chunk)

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout,
            is_causal=True,
            scale=None,
        )

        # bksd -> bskd -> bsh
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, S, H // self.tp_size)

        return self.out_proj(attn_out)
