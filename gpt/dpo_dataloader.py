import json
import numpy as np
import torch
from tqdm import tqdm
import random
from pathlib import Path
from tokenizer import tokenizer
from multiprocessing import Pool


def _tokenize_messages(tok: tokenizer, msgs: list) -> tuple[list[int], list[int]]:
    # same ChatML rule as sft_dataloader._tokenize_conv, but takes a message list
    # directly: <|im_start|>role\n + content + <|im_end|> + \n
    all_ids = []
    token_mask = []

    for msg in msgs:
        role = msg["role"]
        content = msg["content"]

        header_ids = tok.encode(f"<|im_start|>{role}\n")
        all_ids.extend(header_ids)
        token_mask.extend([0] * len(header_ids))

        content_ids = tok.encode(content)
        all_ids.extend(content_ids)
        mask_val = 1 if role == "assistant" else 0
        token_mask.extend([mask_val] * len(content_ids))

        all_ids.append(tok.vocab["<|im_end|>"])
        token_mask.append(mask_val)

        newline_ids = tok.encode("\n")
        all_ids.extend(newline_ids)
        token_mask.extend([0] * len(newline_ids))

    return all_ids, token_mask


class dpo_dataloader:
    def __init__(self, jsonl_path: Path, tok: tokenizer, seq_len: int, batch_size: int):
        self.tok = tok
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pad_id = tok.vocab["<|pad|>"]

        if not jsonl_path.exists():
            print(f"[DPO Dataloader] {jsonl_path} does not exist")
            raise FileNotFoundError(f"{jsonl_path} does not exist")

        self.samples = self._load_samples(jsonl_path)

    def _cache_path(self, jsonl_path: Path) -> Path:
        return jsonl_path.with_suffix(".npz")

    def _load_samples(self, jsonl_path: Path):
        cache_path = self._cache_path(jsonl_path)
        if cache_path.exists() and cache_path.stat().st_mtime >= jsonl_path.stat().st_mtime:
            data = np.load(cache_path)
            if data["tok_fp"].item() == self.tok.fingerprint():
                print(f"[Info] loading cached samples from {cache_path}")
                return [
                    (data["chosen_ids"][i].tolist(), data["chosen_mask"][i].tolist(),
                     data["rejected_ids"][i].tolist(), data["rejected_mask"][i].tolist())
                    for i in range(len(data["chosen_ids"]))
                ]
            print(f"[Info] cache {cache_path} tokenizer mismatch, re-encoding")

        print(f"[Info] loading {jsonl_path}")
        raw = []
        for line in jsonl_path.open():
            line = line.strip()
            if not line:
                continue
            raw.append(json.loads(line))

        tokenized = [self._tokenize_sample(s) for s in tqdm(raw)]

        # truncate to seq_len
        truncated = [
            (c_ids[:self.seq_len], c_mask[:self.seq_len], r_ids[:self.seq_len], r_mask[:self.seq_len])
            for c_ids, c_mask, r_ids, r_mask in tokenized
        ]

        # pad to the longest real sequence (chosen or rejected) so the npz is rectangular
        max_len = max(max(len(c_ids), len(r_ids)) for c_ids, _, r_ids, _ in truncated)
        max_len = max(1, max_len)

        padded = [
            self._pad(c_ids, c_mask, r_ids, r_mask, max_len)
            for c_ids, c_mask, r_ids, r_mask in truncated
        ]

        chosen_ids_arr = np.array([c_ids for c_ids, _, _, _ in padded], dtype=np.uint32)
        chosen_mask_arr = np.array([c_mask for _, c_mask, _, _ in padded], dtype=np.uint8)
        rejected_ids_arr = np.array([r_ids for _, _, r_ids, _ in padded], dtype=np.uint32)
        rejected_mask_arr = np.array([r_mask for _, _, _, r_mask in padded], dtype=np.uint8)

        np.savez(cache_path,
                 chosen_ids=chosen_ids_arr, chosen_mask=chosen_mask_arr,
                 rejected_ids=rejected_ids_arr, rejected_mask=rejected_mask_arr,
                 tok_fp=self.tok.fingerprint())
        print(f"[Info] cached samples to {cache_path}")
        return padded

    def _tokenize_sample(self, sample: dict):
        prompt_ids, _ = _tokenize_messages(self.tok, sample["prompt"])
        chosen_ids, chosen_mask = _tokenize_messages(self.tok, sample["chosen"])
        rejected_ids, rejected_mask = _tokenize_messages(self.tok, sample["rejected"])

        # full sequences: prompt prefix (mask 0) + completion (mask 1)
        chosen_seq = prompt_ids + chosen_ids
        chosen_seq_mask = [0] * len(prompt_ids) + chosen_mask
        rejected_seq = prompt_ids + rejected_ids
        rejected_seq_mask = [0] * len(prompt_ids) + rejected_mask
        return chosen_seq, chosen_seq_mask, rejected_seq, rejected_seq_mask

    def _pad(self, c_ids, c_mask, r_ids, r_mask, max_len: int):
        return (
            c_ids + [self.pad_id] * (max_len - len(c_ids)),
            c_mask + [0] * (max_len - len(c_mask)),
            r_ids + [self.pad_id] * (max_len - len(r_ids)),
            r_mask + [0] * (max_len - len(r_mask)),
        )

    def __iter__(self):
        if self.batch_size > len(self.samples):
            print(f"[DPO Dataloader] batch size {self.batch_size} is larger than the number of samples {len(self.samples)}")
            raise ValueError(f"invalid data size")

        while True:
            perm = np.random.permutation(len(self.samples))
            for start in range(0, len(perm), self.batch_size):
                end = start + self.batch_size
                if end > len(perm):
                    break

                batch_c_ids = []
                batch_c_mask = []
                batch_r_ids = []
                batch_r_mask = []
                for idx in perm[start:end]:
                    c_ids, c_mask, r_ids, r_mask = self.samples[idx]
                    batch_c_ids.append(c_ids)
                    batch_c_mask.append(c_mask)
                    batch_r_ids.append(r_ids)
                    batch_r_mask.append(r_mask)

                yield (torch.tensor(batch_c_ids, dtype=torch.long),
                       torch.tensor(batch_c_mask, dtype=torch.bool),
                       torch.tensor(batch_r_ids, dtype=torch.long),
                       torch.tensor(batch_r_mask, dtype=torch.bool))


class dpo_dataloader_manager:
    def __init__(self, bin_dir: Path, tok: tokenizer, seq_len: int, batch_size: int):
        self.tok = tok
        self.bin_dir = bin_dir
        self._reload_bin_files()

        self.seq_len = seq_len
        self.batch_size = batch_size

        self.current_file_index = 0
        self._change_loader()

    def _reload_bin_files(self):
        self.bin_files = list(self.bin_dir.glob("*.npz"))
        random.shuffle(self.bin_files)
        print("[Info] [dpo_dataloader_manager] load", len(self.bin_files), "dpo-training data files")

    def _change_loader(self):
        current_file = self.bin_files[self.current_file_index]
        current_file_json = Path(current_file).parent / (Path(current_file).stem + ".jsonl")
        print("[Info] [dpo_dataloader_manager] using", current_file)
        self.current_dataloader = dpo_dataloader(current_file_json, self.tok, self.seq_len, self.batch_size)
        self.iter_count = 0

    def __iter__(self):
        reload_countdown = 10
        while True:
            for batch in self.current_dataloader:
                if self.iter_count >= 256:
                    break
                self.iter_count += 1
                yield batch

            self.current_file_index += 1
            if self.current_file_index >= len(self.bin_files):
                self.current_file_index = 0
            self._change_loader()

            reload_countdown -= 1
            if reload_countdown <= 0:
                self._reload_bin_files()
                reload_countdown = 10


def json_to_npz(tok_json: Path, input: Path, output: Path):
    if not input.exists():
        tqdm.write(f"[Info] {input} not exists, skip")
        return
    if len(input.read_bytes()) == 0:
        tqdm.write(f"[Info] {input} is empty, skip")
        return
    if output.exists():
        tqdm.write(f"[Info] {output} exists, skip")
        return

    tok = tokenizer(tok_json)
    try:
        from lib.model_config import model_config
        cfg = model_config()
        dpo_dataloader(input, tok, cfg.max_seq_len, 1)
    except ImportError:
        tqdm.write("[Error] import error")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Quetzalcoatlus GPT-2 dpo-dataloader")
    ap.add_argument("--prepare", action="store_true", help="prepare dpo data")
    ap.add_argument("-j", "--jobs", type=int, default=4, help="number of jobs")
    args = ap.parse_args()

    if not args.prepare:
        tok = tokenizer(Path("data/tokenizer.json"))
        dl = dpo_dataloader(Path("data/rlhf_training_data/dpo.jsonl"), tok, seq_len=128, batch_size=2)

        for i, (c_ids, c_mask, r_ids, r_mask) in enumerate(dl):
            print(f"batch {i}: chosen {c_ids.shape}, rejected {r_ids.shape}")
            print(f"  chosen_ids[0]: {c_ids[0].tolist()}")
            print(f"  chosen_mask[0]: {c_mask[0].tolist()}")
            print(f"  completion tokens in chosen[0]: {c_mask[0].sum().item()}")
            if i >= 2:
                break

    if args.prepare:
        print("====================== DPO ======================")
        data_dir = Path("data")
        json_data_dir = data_dir / "rlhf_training_data"
        if not data_dir.exists():
            data_dir.mkdir()
        if not json_data_dir.exists():
            json_data_dir.mkdir()
        print("[Info] Data directory:", json_data_dir)
        print("[Info] Start converting with", args.jobs, "jobs")

        files = list(json_data_dir.glob("*.jsonl"))
        tasks = [[data_dir / "tokenizer.json", f, json_data_dir / f"{f.stem}.npz"] for f in files]
        pool = Pool(processes=args.jobs)
        try:
            pool.starmap(json_to_npz, tasks)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise
        print("====================== DPO[DONE] ================")
