"""Prototype training loop for heterogeneous attention routing."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from .attention import HeteroAttention
from .routing import HeadRouter


class HeteroDocumentModel(nn.Module):
    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 768):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.router = HeadRouter(hidden_dim)
        self.attention = HeteroAttention(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, input_ids, text_mask, image_mask):
        x = self.embed(input_ids)
        routing = self.router(x.mean(dim=1))
        x = self.attention(x, text_mask, image_mask) * routing.mean(dim=-1, keepdim=True)
        return self.classifier(x[:, 0, :])


def train(epochs: int = 2, batch_size: int = 4) -> None:
    model = HeteroDocumentModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        ids = torch.randint(0, 32000, (batch_size, 512))
        text_mask = torch.ones(batch_size, 512)
        image_mask = torch.zeros(batch_size, 512)
        labels = torch.randint(0, 2, (batch_size,))
        optimizer.zero_grad()
        logits = model(ids, text_mask, image_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        print(f"epoch={epoch} loss={loss.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    train(parser.parse_args().epochs)
