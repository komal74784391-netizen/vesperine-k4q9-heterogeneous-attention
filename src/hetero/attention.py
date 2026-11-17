"""Heterogeneous attention heads."""

from __future__ import annotations

import torch
import torch.nn as nn


class HeteroAttention(nn.Module):
    def __init__(self, hidden_dim: int = 768, heads: int = 12, head_dim: int = 64):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.query = nn.Linear(hidden_dim, heads * head_dim)
        self.key = nn.Linear(hidden_dim, heads * head_dim)
        self.value = nn.Linear(hidden_dim, heads * head_dim)
        self.out = nn.Linear(heads * head_dim, hidden_dim)

    def forward(self, x: torch.Tensor, text_mask: torch.Tensor, image_mask: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.query(x).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        scores = scores + text_mask[:, None, :, None].float() + image_mask[:, None, None, :].float()
        weights = torch.softmax(scores, dim=-1)
        context = (weights @ v).transpose(1, 2).reshape(batch, seq, self.heads * self.head_dim)
        return self.out(context)
