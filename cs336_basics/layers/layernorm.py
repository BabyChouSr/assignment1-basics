import einops
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model).to(device, dtype))

    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)

        norm = torch.einsum("bsd,bsd->bsd", x, x)
        norm = torch.einsum("...d->...", norm)
        norm = (1 / self.d_model * norm + self.eps) ** 0.5
        norm = einops.rearrange(norm, "b s -> b s 1")
        result = x / norm * self.weight
        return result.to(dtype=in_dtype)