import torch
import torch.nn as nn
import numpy as np
import math

VOCABULARY_SIZE = 100
WORD_VEC_LEN = 256 * 2

def xavier(shape: tuple):
    return nn.Parameter(nn.init.xavier_uniform_(torch.empty(shape)))

class single_head_attn(nn.Module):
    def __init__(self, d_model: int, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.Wq = xavier((d_model, d_model))
        self.Wk = xavier((d_model, d_model))
        self.Wv = xavier((d_model, d_model))
        self.dropout_attn = nn.Dropout(dropout)

    def forward(self, x):
        # x: (seq_len, d_model)
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        Q = x @ self.Wq.T # (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
        K = x @ self.Wk.T
        V = x @ self.Wv.T

        # (seq_len, seq_len)
        scores = Q @ K.T / math.sqrt(self.d_model)
        # row-wise softmax
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout_attn(attn)
        # (seq_len, seq_len) @ (seq_len, d_model) -> (seq_len, d_model)
        out = attn @ V
        return out

class multi_head_attn(nn.Module):
    def __init__(self, d_model: int, head: int, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.head = head
        if self.d_model % self.head != 0:
            raise ValueError(f"{self.d_model} is not divisible by {self.head}")
        self.Wq = xavier((d_model, d_model))
        self.Wk = xavier((d_model, d_model))
        self.Wv = xavier((d_model, d_model))
        self.Wo = xavier((d_model, d_model))
        self.dropout_attn = nn.Dropout(dropout)

    def forward(self, x):
        # x: (seq_len, d_model)
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        Q = x @ self.Wq.T # (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
        K = x @ self.Wk.T
        V = x @ self.Wv.T

        d_k = self.d_model // self.head
        # (head, seq_len, d_k)
        seq_len = x.shape[-2]
        Q = Q.view(seq_len, self.head, d_k).transpose(0, 1)
        K = K.view(seq_len, self.head, d_k).transpose(0, 1)
        V = V.view(seq_len, self.head, d_k).transpose(0, 1)

        # (head, seq_len, seq_len)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        # row-wise softmax
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout_attn(attn)
        # (head, seq_len, seq_len) @ (head, seq_len, d_k) -> (head, seq_len, d_k)
        out = attn @ V

        # (seq_len, head, d_k) -> (seq_len, d_model)
        out = out.transpose(0, 1).reshape(seq_len, self.d_model)
        # (seq_len, d_model) * (d_model, d_model) -> (seq_len, d_model)
        out = out @ self.Wo.T
        return out

if __name__ == "__main__":
    rng = np.random.default_rng(seed=114514)

    voca = rng.uniform(-1, 1, (VOCABULARY_SIZE, WORD_VEC_LEN))

    # (5, WORD_VEC_LEN)
    token_ids = [3, 17, 42, 8, 99]
    x = np.stack([voca[i] for i in token_ids])

    attn = single_head_attn(WORD_VEC_LEN)
    out = attn.forward(x)
    print("output shape (single):", out.shape) # (5, WORD_VEC_LEN)
    print(out)

    mattn = multi_head_attn(WORD_VEC_LEN, 16)
    out = mattn.forward(x)
    print("output shape (multi):", out.shape) # (5, WORD_VEC_LEN)
    print(out)