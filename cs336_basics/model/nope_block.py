import torch
import torch.nn as nn

from cs336_basics.layers.attention import AttentionNoRope
from cs336_basics.layers.layernorm import RMSNorm
from cs336_basics.layers.activations import SwiGLU

class Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, device=None, dtype=None):
        super().__init__()

        self.attn = AttentionNoRope(d_model, num_heads, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape

        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        
        return x