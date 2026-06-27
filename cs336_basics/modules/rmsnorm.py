import torch
from einops import reduce
from jaxtyping import Float
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.empty((self.d_model), device=device, dtype=dtype))
        self.reset_parameters()

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_squared = reduce((x**2.0), "... d_model -> ... 1", "mean") + self.eps
        rms = mean_squared**0.5
        result = (x / rms) * self.g
        return result.to(in_dtype)

    def reset_parameters(self):
        torch.nn.init.ones_(self.g)
