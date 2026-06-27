import math

import torch
from einops import einsum
from torch import nn


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.b = nn.Parameter(torch.empty((out_features), device=device, dtype=dtype))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Wx = einsum(x, self.W, "... in_features, out_features in_features -> ... out_features")
        return Wx + self.b

    def reset_parameters(self):
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        torch.nn.init.zeros_(self.b)
