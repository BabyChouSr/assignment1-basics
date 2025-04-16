import torch
import torch.nn as nn

from cs336_basics.layers.attention import MultiheadSelfAttention
from cs336_basics.layers.layernorm import RMSNorm
from cs336_basics.layers.activations import SwiGLU

class Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len, device=None, dtype=None):
        super().__init__()

        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape

        token_positions = torch.arange(0, s, device=x.device)
        x = x + self.attn(x, token_positions)
        x = x + self.ffn(x)
        
        return x


