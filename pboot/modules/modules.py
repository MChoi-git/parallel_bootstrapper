"""Contains a wide collection of modules for use in transformers.
"""
from argparse import Namespace
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, config: Namespace, base: int = 10000):
        super().__init__()
        self.head_dim = config.model_dim // config.nheads
        self.max_seq_len = config.max_seq_len
        self.base = base

        # Frequency states for initial sequence length
        inv_freq, cos, sin = self._update_freqs(self.max_seq_len)
        self.register_buffer("inv_freq", inv_freq)
        self.cos_cache = cos
        self.sin_cache = sin

    def _update_freqs(self, new_seq_len: int):
        inverse_freq = 1. / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))

        positions = torch.arange(new_seq_len).float()
        freqs = torch.einsum("i,j->ij", positions, inverse_freq)
        embed = torch.cat((freqs, freqs), dim=-1)

        cos = embed.cos()[None, :, None, :]
        sin = embed.sin()[None, :, None, :]

        return inverse_freq, cos, sin

    def forward(self, inputs: Tensor):
        # Assume (batch, seq, heads, hdim)
        seq_len = inputs.shape[1]

        # Update frequencies on larger sequence length
        if seq_len > self.max_seq_len:
            self.inv_feq, self.cos_cache, self.sin_cache = self._update_freqs(seq_len)
            self.max_seq_len = seq_len

        return (
            self.cos_cache[:, : seq_len, ...].to(inputs.device),
            self.sin_cache[:, : seq_len, ...].to(inputs.device),
        )


def apply_rotary_position_embeddings(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
    def rotate_half(x: Tensor):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2: ]
        return torch.cat((-x2, x1), dim=-1)

    return (
        ((q * cos) + (rotate_half(q) * sin)).type_as(q),
        ((k * cos) + (rotate_half(k) * sin)).type_as(k),
    )


class VocabEmbedding(nn.Module):
    def __init__(self, config: Namespace):
        super().__init__()
        self.model_dim = config.model_dim
        self.vocab_size = config.vocab_size
        self.dtype = config.dtype
        self.device = config.device

        self.vocab = nn.Embedding(self.vocab_size, self.model_dim, dtype=self.dtype, device=self.device)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.vocab(inputs)


class MLP(nn.Module):
    def __init__(self, config: Namespace):
        super().__init__()
        self.model_dim = config.model_dim
        self.hidden_dim = int(self.model_dim * config.mlp_expansion_scale)
        self.use_bias = config.use_linear_bias
        self.dtype = config.dtype
        self.device = config.device

        self.fc1 = nn.Linear(self.model_dim, self.hidden_dim, bias=self.use_bias, dtype=self.dtype, device=self.device)
        self.act_fn = torch.nn.functional.relu
        self.fc2 = nn.Linear(self.hidden_dim, self.model_dim, bias=self.use_bias, dtype=self.dtype, device=self.device)

    def forward(self, inputs):
        return self.fc2(self.act_fn(self.fc1(inputs)))


class MHA(nn.Module):
    def __init__(self, config: Namespace):
        super().__init__()
        self.model_dim = config.model_dim
        self.nheads = config.nheads
        self.attention_dropout = config.attention_dropout
        self.dtype = config.dtype
        self.device = config.device

        self.apply_rope_fn = apply_rotary_position_embeddings
        self.qkv_proj = nn.Linear(self.model_dim, 3 * self.model_dim, dtype=self.dtype, device=self.device)
        self.out_proj = nn.Linear(self.model_dim, self.model_dim, dtype=self.dtype, device=self.device)

    def forward(self, inputs: Tensor, cos: Tensor, sin: Tensor, attention_mask: Optional[Tensor] = None):
        B, S, H = inputs.shape

        q, k, v = torch.chunk(self.qkv_proj(inputs), 3, dim=-1)

        assert H % self.nheads == 0

        q = q.reshape(B, S, self.nheads, H // self.nheads)
        k = k.reshape(B, S, self.nheads, H // self.nheads)
        v = v.reshape(B, S, self.nheads, H // self.nheads)

        q, k = self.apply_rope_fn(q, k, cos, sin)

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout,
            is_causal=True,
            scale=None,
        )

        attn_out = attn_out.reshape(B, S, H)

        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, config: Namespace):
        super().__init__()
        self.model_dim = config.model_dim
        self.mha_dropout = config.mha_dropout
        self.mlp_dropout = config.mlp_dropout
        self.device = config.device

        self.mha = MHA(config)
        self.mlp = MLP(config)

        # LayerNorms always in fp32
        self.ln1 = nn.LayerNorm(self.model_dim, device=self.device)
        self.ln2 = nn.LayerNorm(self.model_dim, device=self.device)

        self.mha_dropout = nn.Dropout(p=self.mha_dropout)
        self.mlp_dropout = nn.Dropout(p=self.mlp_dropout)

    def forward(self, inputs: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        res1 = inputs
        mha_out = self.ln1(inputs.float()).type_as(inputs)
        mha_out = self.mha(mha_out, cos, sin)
        mha_out = self.mha_dropout(mha_out) + res1

        res2 = mha_out
        mlp_out = self.ln2(mha_out.float()).type_as(inputs)
        mlp_out = self.mlp(mlp_out)
        mlp_out = self.mlp_dropout(mlp_out) + res2

        return mlp_out
