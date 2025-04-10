import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        std = (2 / (in_features + out_features)) ** 0.5
        weight_init = nn.init.trunc_normal_(torch.randn(in_features, out_features) * std, a = -3 * std, b = 3 * std).to(device=device, dtype=dtype)
        self.weight = nn.Parameter(weight_init)

    def forward(self, x: torch.Tensor):
        # o, i is output, input shape since we store W_transpose
        x = torch.einsum("b s i, o i -> b s o", x, self.weight)
        return x