import transformer
import torch
from torch import Tensor
import torch.nn as nn

class gpt(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 head: int,
                 n_layers: int,
                 max_seq_len: int,
                 dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            transformer.block(d_model, head, max_seq_len, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, x: Tensor):
        # x (batch, seq_len) token ids
        h = self.tok_emb(x) # (batch, seq_len, d_model)
        for blk in self.blocks:
            h = blk(h)

        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits

if __name__ == "__main__":
    VOCABULARY_SIZE = 100
    WORD_VEC_LEN = 704
    HEAD = 11
    LAYER = 30
    MAX_SEQ_LEN = 2048

    model = gpt(VOCABULARY_SIZE, WORD_VEC_LEN, HEAD, LAYER, MAX_SEQ_LEN)

    token_ids = torch.randint(0, VOCABULARY_SIZE, (1, 5))
    out = model.forward(token_ids)
    print("output shape (multi):", out.shape) # (5, WORD_VEC_LEN)
    print(out)