import torch
import torch.nn as nn
from cs336_basics.layers.linear import Linear

class Silu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(x)

class SiluFFN(nn.Module):
    def __init__(self, d_model: int, d_ff:int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_ff, d_model, device, dtype)
        self.w2 = Linear(d_model, d_ff, device, dtype)
        self.silu = Silu()
    
    def forward(self, x: torch.Tensor):
        x = self.w2(self.silu(self.w1(x)))
        return x
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_ff, d_model, device, dtype)
        self.w2 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_ff, d_model, device, dtype)
        self.silu = Silu()

    def forward(self, x: torch.Tensor):
        w1_x = self.w1(x)
        silu_x = self.silu(w1_x)
        w3_x = self.w3(x)
        x = torch.einsum("...,...->...", silu_x, w3_x)
        x = self.w2(x)
        return x
