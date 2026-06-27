import math

import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.E = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.E[x]

    def reset_parameters(self):
        std = math.sqrt(2.0 / (self.num_embeddings + self.embedding_dim))
        torch.nn.init.trunc_normal_(self.E, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
