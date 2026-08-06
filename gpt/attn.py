import torch
from torch import Tensor
import torch.nn as nn
import math

def _rope_cache(attn: nn.Module, max_seq_len: int, d_k: int):
    assert d_k % 2 == 0, "d_model must be even for RoPE"
    # (d_k // 2,)
    freqs = 1 / (10000.0 ** (2 * torch.arange(0, d_k, 2).float() / d_k))
    # (max_seq_len,)
    pos = torch.arange(max_seq_len).float()
    # (max_seq_len, d_k // 2)
    angles = pos[:, None] * freqs[None, :] # equals to torch.outer(pos, freqs)

    attn.register_buffer("cos_cache", torch.cos(angles))
    attn.register_buffer("sin_cache", torch.sin(angles))

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
    def __init__(self, d_model: int, max_seq_len: int, dropout = 0.0):
        super().__init__()
        self.d_model = d_model
        _rope_cache(self, max_seq_len, self.d_model)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.dropout_attn = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (seq_len, d_model)
        Q = self.Wq(x) # (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
        K = self.Wk(x)
        V = self.Wv(x)

        seq_len = x.shape[-2]
        cos = self.cos_cache[:seq_len]
        sin = self.sin_cache[:seq_len]
        Q = _apply_rope(Q, cos, sin)
        K = _apply_rope(K, cos, sin)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        # (seq_len, seq_len)
        scores = Q @ K.T / math.sqrt(self.d_model)
        scores = scores.masked_fill(causal_mask, float('-inf'))
        # row-wise softmax
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout_attn(attn)
        # (seq_len, seq_len) @ (seq_len, d_model) -> (seq_len, d_model)
        out = attn @ V
        return out

class multi_head_attn(nn.Module):
    def __init__(self, d_model: int, head: int, max_seq_len: int, dropout = 0.0):
        super().__init__()
        self.d_model = d_model
        self.head = head
        if self.d_model % self.head != 0:
            raise ValueError(f"{self.d_model} is not divisible by {self.head}")
        self.d_k = self.d_model // self.head
        _rope_cache(self, max_seq_len, self.d_k)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout_attn = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (seq_len, d_model)
        Q = self.Wq(x) # (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
        K = self.Wk(x)
        V = self.Wv(x)

        # (head, seq_len, d_k)
        seq_len = x.shape[-2]
        Q = Q.view(seq_len, self.head, self.d_k).transpose(0, 1)
        K = K.view(seq_len, self.head, self.d_k).transpose(0, 1)
        V = V.view(seq_len, self.head, self.d_k).transpose(0, 1)

        cos = self.cos_cache[:seq_len]
        sin = self.sin_cache[:seq_len]
        Q = _apply_rope(Q, cos, sin)
        K = _apply_rope(K, cos, sin)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        # (head, seq_len, seq_len)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        scores = scores.masked_fill(causal_mask, float('-inf'))
        # row-wise softmax
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout_attn(attn)
        # (head, seq_len, seq_len) @ (head, seq_len, d_k) -> (head, seq_len, d_k)
        out = attn @ V

        # (seq_len, head, d_k) -> (seq_len, d_model)
        out = out.transpose(0, 1).reshape(seq_len, self.d_model)
        # (seq_len, d_model) * (d_model, d_model) -> (seq_len, d_model)
        out = self.Wo(out)
        return out

if __name__ == "__main__":
    import numpy as np

    VOCABULARY_SIZE = 100
    WORD_VEC_LEN = 704
    HEAD = 11
    MAX_SEQ_LEN = 2048

    rng = np.random.default_rng(seed=114514)

    voca = rng.uniform(-1, 1, (VOCABULARY_SIZE, WORD_VEC_LEN))

    # (5, WORD_VEC_LEN)
    token_ids = [3, 17, 42, 8, 99]
    x = np.stack([voca[i] for i in token_ids])
    x = torch.from_numpy(x).float()

    attn = single_head_attn(WORD_VEC_LEN, MAX_SEQ_LEN)
    out = attn.forward(x)
    print("output shape (single):", out.shape) # (5, WORD_VEC_LEN)
    print(out)

    mattn = multi_head_attn(WORD_VEC_LEN, HEAD, MAX_SEQ_LEN)
    out = mattn.forward(x)
    print("output shape (multi):", out.shape) # (5, WORD_VEC_LEN)
    print(out)
