import json
import numpy as np
import torch
import tqdm
from pathlib import Path
from tokenizer import tokenizer

class sft_dataloader:
    def __init__(self, jsonl_path: Path, tok: tokenizer, seq_len: int, batch_size: int):
        self.tok = tok
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pad_id = tok.vocab["<|pad|>"]
        self.im_start_id = tok.vocab["<|im_start|>"]
        self.im_end_id = tok.vocab["<|im_end|>"]

        if not jsonl_path.exists():
            print(f"[SFT Dataloader] {jsonl_path} does not exist")
            raise FileNotFoundError(f"{jsonl_path} does not exist")

        self.windows = self._load_windows(jsonl_path)

    def _cache_path(self, jsonl_path: Path) -> Path:
        return jsonl_path.with_suffix(f".sft.npz")

    def _load_windows(self, jsonl_path: Path):
        cache_path = self._cache_path(jsonl_path)
        if cache_path.exists() and cache_path.stat().st_mtime >= jsonl_path.stat().st_mtime:
            data = np.load(cache_path)
            if data["tok_fp"].item() == self.tok.fingerprint():
                print(f"[Info] loading cached windows from {cache_path}")
                return [(ids.tolist(), mask.tolist())
                        for ids, mask in zip(data["ids"], data["mask"])]
            print(f"[Info] cache {cache_path} tokenizer mismatch, re-encoding")

        print(f"[Info] loading {jsonl_path}")
        # load conversations from jsonl file, in line
        conversations = []
        for line in jsonl_path.open():
            line = line.strip()
            if not line:
                continue
            conversations.append(json.loads(line))

        # tokenize conversations and pack into windows
        tokenized = [self._tokenize_conv(c) for c in tqdm.tqdm(conversations)]
        windows = self._pack(tokenized)
        print(f"[Info] packed {len(windows)} windows")

        ids_arr = np.array([ids for ids, _ in windows], dtype=np.uint32)
        mask_arr = np.array([mask for _, mask in windows], dtype=np.uint8)
        np.savez(cache_path, ids=ids_arr, mask=mask_arr, tok_fp=self.tok.fingerprint())
        print(f"[Info] cached windows to {cache_path}")
        return windows

    def _tokenize_conv(self, conv) -> tuple[list[int], list[int]]:
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
            token_mask.append(mask_val)

            newline_ids = self.tok.encode("\n")
            all_ids.extend(newline_ids)
            token_mask.extend([0] * len(newline_ids))

        return all_ids, token_mask

    def _pack(self, tokenized: list[tuple[list[int], list[int]]]):
        windows = []
        cur_ids = []
        cur_mask = []

        for ids, mask in tokenized:
            # truncate if too long
            if len(ids) > self.seq_len:
                ids = ids[:self.seq_len]
                mask = mask[:self.seq_len]

            # current window is full, insert and clear temporary buffers
            if len(cur_ids) + len(ids) > self.seq_len:
                windows.append((cur_ids, cur_mask))
                cur_ids = []
                cur_mask = []

            cur_ids.extend(ids)
            cur_mask.extend(mask)

        # pad the last window
        if cur_ids:
            windows.append((cur_ids, cur_mask))

        padded = []
        for ids, mask in windows:
            pad_len = self.seq_len - len(ids)
            # pad with <|pad|> to align with the seq_len
            padded.append((
                ids + [self.pad_id] * pad_len,
                mask + [0] * pad_len,
            ))

        return padded

    def __iter__(self):
        if self.batch_size > len(self.windows):
            print(f"[SFT Dataloader] batch size {self.batch_size} is larger than the number of windows {len(self.windows)}")
            raise ValueError(f"invalid data size")

        while True:
            # randomly permute the windows
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
    tok = tokenizer(Path("data/tokenizer.json"))
    dl = sft_dataloader(Path("data/SFT.jsonl"), tok, seq_len=128, batch_size=2)

    for i, (inp, tgt) in enumerate(dl):
        print(f"batch {i}: input {inp.shape}, targets {tgt.shape}")
        print(f"  input[0]: {inp[0].tolist()}")
        print(f"  target[0]: {tgt[0].tolist()}")
        print(f"  trainable tokens in target[0]: {(tgt[0] != -100).sum().item()}")
        if i >= 2:
            break
