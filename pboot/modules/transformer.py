"""Contains top-level modules for various types of transformers.
"""
from argparse import Namespace
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from pboot.modules import (
    TransformerBlock,
    RotaryPositionEmbedding,
    VocabEmbedding,
)


def _ce_loss(logits, targets):
    logits = logits.view(-1, logits.shape[-1])
    targets = targets.view(-1)
    return torch.nn.functional.cross_entropy(logits, targets)


class TransformerLM(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.model = Transformer(*args, **kwargs)
        self.loss_fn = _ce_loss

    def forward(self, batch: Tensor, *args, **kwargs):
        logits = self.model(batch, *args, **kwargs)

        with torch.no_grad():
            targets = batch[..., 1:].contiguous()

        logits = logits[..., :-1, :].contiguous()

        loss = self.loss_fn(logits, targets)

        return loss


class Transformer(nn.Module):
    def __init__(self, config: Namespace):
        super().__init__()
        assert config.model_dim % config.nheads == 0
        self.nlayers = config.nlayers
        self.model_dim = config.model_dim
        self.vocab_size = config.vocab_size
        self.device = config.device
        self.dtype = config.dtype

        # Preprocess
        self.vocab_embeds = VocabEmbedding(config)

        self.rope = RotaryPositionEmbedding(config)

        # Core
        self.layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.nlayers)
        ])

        # Postprocess
        self.out_ln = nn.LayerNorm(self.model_dim, device=self.device)
        self.out_proj = nn.Linear(self.model_dim, self.vocab_size, device=self.device, dtype=self.dtype)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.vocab_embeds(inputs)
        cos, sin = self.rope(hidden)

        for layer in self.layers:
            hidden = layer(hidden, cos, sin)

        out = self.out_ln(hidden.float()).to(self.dtype)
        out = self.out_proj(out)

        return out
