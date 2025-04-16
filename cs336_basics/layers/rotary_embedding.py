import torch
import torch.nn as nn

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        
        self.d_k = d_k

        k = torch.arange(0, d_k // 2, device=device)
        inv_freqs = 1 / theta ** (2 * k / d_k)
        positions = torch.arange(0, max_seq_len, device=device)
        freqs = torch.einsum("i,j->ij", positions, inv_freqs)

        # s, d //2
        cos = torch.cos(freqs)
        # s, d //2
        sin = torch.sin(freqs)

        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)
        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        x1 = x[..., ::2] # b, s, d // 2 even positions
        x2 = x[..., 1::2] # b, s, d // 2 odd positions


        sin_embed = self.sin[token_positions, ...]
        cos_embed = self.cos[token_positions, ...]
        even = cos_embed * x1 - sin_embed * x2 # b, s, d // 2 * b, s, d // 2 -> b, s, d // 2
        odd = sin_embed * x1 + cos_embed * x2  # b, s, d // 2 * b, s, d // 2 -> b, s, d // 2
        
        # b, s, d // 2, 2
        rotated_x = torch.stack([even, odd], dim=-1)
        return rotated_x.view_as(x)