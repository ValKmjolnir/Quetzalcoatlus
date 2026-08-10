import json
import numpy as np
import torch
from pathlib import Path


class sft_dataloader:
    def __init__(self, jsonl_path: Path, tok, seq_len: int, batch_size: int):
        self.tok = tok
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pad_id = tok.vocab["<|pad|>"]
        self.im_start_id = tok.vocab["<|im_start|>"]
        self.im_end_id = tok.vocab["<|im_end|>"]

        conversations = []
        for line in jsonl_path.open():
            line = line.strip()
            if not line:
                continue
            conversations.append(json.loads(line))

        tokenized = [self._tokenize_conv(c) for c in conversations]
        self.windows = self._pack(tokenized)

    def _tokenize_conv(self, conv):
        all_ids = []
        token_mask = []

        for msg in conv["messages"]:
            role = msg["role"]
            content = msg["content"]

            header_ids = self.tok.encode(f"<|im_start|>{role}\n")
            all_ids.extend(header_ids)
            token_mask.extend([0] * len(header_ids))

            content_ids = self.tok.encode(content)
            all_ids.extend(content_ids)
            mask_val = 1 if role == "assistant" else 0
            token_mask.extend([mask_val] * len(content_ids))

            all_ids.append(self.im_end_id)
            token_mask.append(1 if role == "assistant" else 0)

            newline_ids = self.tok.encode("\n")
            all_ids.extend(newline_ids)
            token_mask.extend([0] * len(newline_ids))

        return all_ids, token_mask

    def _pack(self, tokenized):
        windows = []
        cur_ids = []
        cur_mask = []

        for ids, mask in tokenized:
            if len(ids) > self.seq_len:
                ids = ids[:self.seq_len]
                mask = mask[:self.seq_len]

            if len(cur_ids) + len(ids) > self.seq_len:
                windows.append((cur_ids, cur_mask))
                cur_ids = []
                cur_mask = []

            cur_ids.extend(ids)
            cur_mask.extend(mask)

        if cur_ids:
            windows.append((cur_ids, cur_mask))

        padded = []
        for ids, mask in windows:
            pad_len = self.seq_len - len(ids)
            padded.append((
                ids + [self.pad_id] * pad_len,
                mask + [0] * pad_len,
            ))

        return padded

    def __iter__(self):
        while True:
            perm = np.random.permutation(len(self.windows))
            for start in range(0, len(perm), self.batch_size):
                end = start + self.batch_size
                if end > len(perm):
                    break

                batch_inputs = []
                batch_targets = []
                for idx in perm[start:end]:
                    ids, token_mask = self.windows[idx]

                    input_ids = torch.tensor(ids, dtype=torch.long)
                    targets = torch.tensor(
                        ids[1:] + [self.pad_id], dtype=torch.long)

                    target_mask = token_mask[1:] + [0]
                    mask_tensor = torch.tensor(target_mask, dtype=torch.bool)
                    targets[~mask_tensor] = -100

                    batch_inputs.append(input_ids)
                    batch_targets.append(targets)

                yield torch.stack(batch_inputs), torch.stack(batch_targets)


if __name__ == "__main__":
    from tokenizer import tokenizer

    tok = tokenizer(Path("tokenizer.json"))
    dl = sft_dataloader(Path("data/SFT.jsonl"), tok, seq_len=128, batch_size=2)

    for i, (inp, tgt) in enumerate(dl):
        print(f"batch {i}: input {inp.shape}, targets {tgt.shape}")
        print(f"  input[0]: {inp[0].tolist()}")
        print(f"  target[0]: {tgt[0].tolist()}")
        print(f"  trainable tokens in target[0]: {(tgt[0] != -100).sum().item()}")
        if i >= 2:
            break
