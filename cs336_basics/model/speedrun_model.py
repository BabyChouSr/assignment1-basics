import torch
import torch.nn as nn
from cs336_basics.layers.layernorm import RMSNorm
from cs336_basics.layers.linear import Linear
from cs336_basics.layers.embedding import Embedding
from cs336_basics.layers.activations import SwiGLU

import einops
import torch.nn.functional as F
from cs336_basics.layers.rotary_embedding import RotaryPositionEmbedding
from cs336_basics.layers.softmax import softmax

def scaled_dot_product_attention(q, k, v, attention_mask):
    d_k = q.shape[-1]
    scores = torch.einsum("b ... s d, b ... t d->b ... s t", q, k) / (d_k ** 0.5)
    scores = scores.masked_fill(attention_mask == 0, float('-inf'))
    probabilities = softmax(scores, i=-1)
    values = torch.einsum("b ... t s, b ... s d-> b ... t d", probabilities, v)
    return values

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        self.d_k = self.d_v = d_model // num_heads
        self.num_heads = num_heads

        key_dimension = int(self.d_k * num_heads)
        value_dimension = int(self.d_v * num_heads)

        self.q_proj = Linear(key_dimension, d_model, device, dtype)
        self.k_proj = Linear(key_dimension, d_model, device, dtype)
        self.v_proj = Linear(value_dimension, d_model, device, dtype)
        self.o_proj = Linear(d_model, value_dimension, device, dtype)

        # self.q_norm = RMSNorm(d_model, device, dtype)
        # self.k_norm = RMSNorm(d_model, device, dtype)

        self.rope = RotaryPositionEmbedding(theta, self.d_k, max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        # ... s, d_in
        seq_len = x.shape[-2]
        q = self.q_proj(x)
        q = einops.rearrange(q, "... s (h d) -> ... h s d", h=self.num_heads)
        q = self.rope(q, token_positions)
        k = self.k_proj(x)
        k = einops.rearrange(k, "... s (h d) -> ... h s d", h=self.num_heads)
        k = self.rope(k, token_positions)
        v = self.v_proj(x)
        v = einops.rearrange(v, "... s (h d) -> ... h s d", h=self.num_heads)

        # Create causal attention mask where each token can only attend to itself and previous tokens
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.float))

        values = scaled_dot_product_attention(q, k, v, mask)
        values = einops.rearrange(values, "... h s d -> ... s (h d)", h=self.num_heads)
        hidden_states = self.o_proj(values)
        return hidden_states

class ReluSquaredFFN(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()

        self.w1 = Linear(d_ff, d_model, device, dtype)
        self.w2 = Linear(d_model, d_ff, device, dtype)
        self.w2.weight.detach().zero_()

    def forward(self, x: torch.Tensor):
        x = self.w1(x)
        x = F.relu(x).square()
        x = self.w2(x)
        return x

class Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len, device=None, dtype=None):
        super().__init__()

        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ffn = ReluSquaredFFN(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape

        token_positions = torch.arange(0, s, device=x.device, dtype=torch.long)
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        
        return x

class SpeedrunModel(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, vocab_size, context_length, num_layers, device=None, dtype=None):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([Block(d_model, num_heads, d_ff, theta, context_length, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(vocab_size, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)

        x = self.lm_head(self.ln_final(x))
        return x
