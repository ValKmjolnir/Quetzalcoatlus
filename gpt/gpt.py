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
                 dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            transformer.block(d_model, head, dropout)
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
        # (batch, seq_len, d_model) @ (d_model, vocab_size) -> (batch, seq_len, vocab_size)
        logits = self.lm_head(h)
        return logits

if __name__ == "__main__":
    VOCABULARY_SIZE = 32000 // 8
    WORD_VEC_LEN = 704 // 4
    HEAD = 11
    LAYER = 30
    MAX_SEQ_LEN = 2048

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = gpt(VOCABULARY_SIZE, WORD_VEC_LEN, HEAD, LAYER).to(device)

    token_ids = torch.randint(0, VOCABULARY_SIZE, (1, 50), device=device)
    out = model.forward(token_ids)
    print("output shape (multi):", out.shape) # (5, WORD_VEC_LEN)
    print(out)