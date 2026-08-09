import torch
from torch import Tensor
import torch.nn as nn
import math

def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # input    (..., seq_len, d)
    # cos, sin (seq_len, d // 2)
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]

    for _ in range(x_even.dim() - 2):
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    rot_even = x_even * cos - x_odd * sin
    rot_odd  = x_even * sin + x_odd * cos

    return torch.stack([rot_even, rot_odd], dim=-1).flatten(start_dim=-2)

class single_head_attn(nn.Module):
    def __init__(self, d_model: int, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.dropout_attn = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, d_model)
        Q = self.Wq(x) # (batch, seq_len, d_model) @ (d_model, d_model) -> (batch, seq_len, d_model)
        K = self.Wk(x) # (batch, seq_len, d_model) @ (d_model, d_model) -> (batch, seq_len, d_model)
        V = self.Wv(x) # (batch, seq_len, d_model) @ (d_model, d_model) -> (batch, seq_len, d_model)

        seq_len = x.shape[-2]
        device = x.device
        d_k = self.d_model

        # RoPE cache
        freqs = 1 / (10000.0 ** (2 * torch.arange(0, d_k, 2, device=device).float() / d_k))
        pos = torch.arange(seq_len, device=device).float()
        angles = pos[:, None] * freqs[None, :]
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        Q = _apply_rope(Q, cos, sin)
        K = _apply_rope(K, cos, sin)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        # (batch, seq_len, seq_len)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_model)
        scores = scores.masked_fill(causal_mask, float('-inf'))
        # row-wise softmax
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout_attn(attn)
        # (batch, seq_len, seq_len) @ (seq_len, d_model) -> (batch, seq_len, d_model)
        out = attn @ V
        return out

class multi_head_attn(nn.Module):
    def __init__(self, d_model: int, head: int, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.head = head
        if self.d_model % self.head != 0:
            raise ValueError(f"{self.d_model} is not divisible by {self.head}")
        self.d_k = self.d_model // self.head
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout_attn = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, d_model)
        Q = self.Wq(x) # (batch, seq_len, d_model) @ (d_model, d_model) -> (batch, seq_len, d_model)
        K = self.Wk(x) # (batch, seq_len, d_model) @ (d_model, d_model) -> (batch, seq_len, d_model)
        V = self.Wv(x) # (batch, seq_len, d_model) @ (d_model, d_model) -> (batch, seq_len, d_model)

        seq_len = x.shape[-2]
        device = x.device
        # (batch, seq_len, head, d_k) -> (batch, head, seq_len, d_k)
        Q = Q.reshape(*Q.shape[:-1], self.head, self.d_k).transpose(-2, -3)
        K = K.reshape(*K.shape[:-1], self.head, self.d_k).transpose(-2, -3)
        V = V.reshape(*V.shape[:-1], self.head, self.d_k).transpose(-2, -3)

        # RoPE cache
        freqs = 1 / (10000.0 ** (2 * torch.arange(0, self.d_k, 2, device=device).float() / self.d_k))
        pos = torch.arange(seq_len, device=device).float()
        angles = pos[:, None] * freqs[None, :]
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        Q = _apply_rope(Q, cos, sin)
        K = _apply_rope(K, cos, sin)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        # (batch, head, seq_len, seq_len)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        scores = scores.masked_fill(causal_mask, float('-inf'))
        # row-wise softmax
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout_attn(attn)
        # (batch, head, seq_len, seq_len) @ (batch, head, seq_len, d_k) -> (batch, head, seq_len, d_k)
        out = attn @ V

        # (batch, seq_len, head, d_k) -> (batch, seq_len, d_model)
        out = out.transpose(-2, -3).flatten(start_dim=-2)
        # (batch, seq_len, d_model) * (d_model, d_model) -> (batch, seq_len, d_model)
        out = self.Wo(out)
        return out

if __name__ == "__main__":
    import numpy as np

    VOCABULARY_SIZE = 100
    WORD_VEC_LEN = 704 // 2
    HEAD = 11
    MAX_SEQ_LEN = 2048 // 2

    rng = np.random.default_rng(seed=114514)

    voca = rng.uniform(-1, 1, (VOCABULARY_SIZE, WORD_VEC_LEN))

    # (5, WORD_VEC_LEN)
    token_ids = [3, 17, 42, 8, 99]
    x = np.stack([voca[i] for i in token_ids])
    x = torch.from_numpy(x).float()

    attn = single_head_attn(WORD_VEC_LEN)
    out = attn.forward(x)
    print("output shape (single):", out.shape) # (5, WORD_VEC_LEN)
    print(out)

    mattn = multi_head_attn(WORD_VEC_LEN, HEAD)
    out = mattn.forward(x)
    print("output shape (multi):", out.shape) # (5, WORD_VEC_LEN)
    print(out)
