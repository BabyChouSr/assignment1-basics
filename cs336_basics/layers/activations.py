import torch
import torch.nn as nn

class Silu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(x)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(d_ff, d_model)).to(device, dtype)
        self.w2 = nn.Parameter(torch.randn(d_model, d_ff)).to(device, dtype)
        self.w3 = nn.Parameter(torch.randn(d_ff, d_model)).to(device, dtype)
        self.silu = Silu()

    def forward(self, x: torch.Tensor):
        w1_x = torch.einsum("... d,o d-> ... o", x, self.w1)
        silu_x = self.silu(w1_x)
        w3_x = torch.einsum("... d, o d -> ... o", x, self.w3)
        x = torch.einsum("...,...->...", silu_x, w3_x)
        x = torch.einsum("... o, d o -> ... d", x, self.w2)
        return x
