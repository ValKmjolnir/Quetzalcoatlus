import torch
import torch.nn as nn

from .attn import multi_head_attn

class ffn(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, dropout = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))

class block(nn.Module):
    def __init__(self, d_model: int, head: int, dropout = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = ffn(d_model)
        self.attn = multi_head_attn(d_model, head)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # pre-norm with residual
        x = x + self.dropout(self.attn(self.ln1(x)))
        # pre-norm ffn with residual
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x