"""Token-to-head routing policy."""

from __future__ import annotations

import torch
import torch.nn as nn


class HeadRouter(nn.Module):
    def __init__(self, hidden_dim: int = 768, heads: int = 12, top_heads: int = 4):
        super().__init__()
        self.heads = heads
        self.top_heads = top_heads
        self.policy = nn.Linear(hidden_dim, heads)

    def forward(self, modality_embedding: torch.Tensor) -> torch.Tensor:
        logits = self.policy(modality_embedding)
        mask = torch.zeros_like(logits)
        idx = torch.topk(logits, self.top_heads, dim=-1).indices
        mask.scatter_(-1, idx, 1.0)
        return torch.softmax(logits, dim=-1) * mask
