import torch
import torch.nn as nn
from cs336_basics.layers.layernorm import RMSNorm
from cs336_basics.layers.linear import Linear
from cs336_basics.layers.embedding import Embedding
from cs336_basics.model.no_rms_block import Block


class NoRMSModel(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, vocab_size, context_length, num_layers, device=None, dtype=None):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([Block(d_model, num_heads, d_ff, theta, context_length, device=device, dtype=dtype) for _ in range(num_layers)])
        self.lm_head = Linear(vocab_size, d_model, device=device, dtype=dtype)
        # self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)

        x = self.lm_head(x)
        return x
