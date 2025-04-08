import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(nn.init.trunc_normal_(torch.randn(num_embeddings, embedding_dim), a=-3, b=3)).to(device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]