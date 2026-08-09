import torch
from torch import Tensor
import torch.nn as nn

from attn import multi_head_attn

class swiglu_ffn(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.0):
        super().__init__()
        d_ff = d_ff or int(8 / 3 * d_model) // 64 * 64
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate = nn.functional.silu(self.gate_proj(x))
        up   = self.up_proj(x)
        return self.down_proj(self.dropout(gate * up))

class ffn(nn.Module):
    def __init__(self, d_model: int, d_ff: int|None=None, dropout=0.0):
        super().__init__()
        d_ff = d_ff or (4 * d_model)
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))

class block(nn.Module):
    def __init__(self, d_model: int, head: int, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = swiglu_ffn(d_model)
        self.attn = multi_head_attn(d_model, head)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # pre-norm with residual
        x = x + self.dropout(self.attn(self.ln1(x)))
        # pre-norm ffn with residual
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

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

    trans = block(WORD_VEC_LEN, HEAD, 0)
    out = trans.forward(x)
    print("output shape (multi):", out.shape) # (5, WORD_VEC_LEN)
    print(out)