from dataclasses import dataclass

@dataclass
class model_config:
    d_model: int = 704 // 2
    head: int = 11
    n_layers: int = 30
    max_seq_len: int = 1024