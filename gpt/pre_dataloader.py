import numpy as np
import torch
import random
from pathlib import Path

class pre_dataloader:
    def __init__(self, bin_path: Path, seq_len: int, batch_size: int):
        self.data = np.memmap(bin_path, dtype=np.uint32, mode='r')
        self.seq_len = seq_len
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __iter__(self):
        while True:
            starts = np.random.randint(
                0, len(self.data) - self.seq_len - 1,
                size=self.batch_size)
            inputs = np.stack([self.data[s : s + self.seq_len] for s in starts])
            targets = np.stack([self.data[s + 1 : s + self.seq_len + 1] for s in starts])

            yield torch.from_numpy(inputs).long(), torch.from_numpy(targets).long()

class pre_dataloader_manager:
    def __init__(self, bin_dir: Path, seq_len: int, batch_size: int):
        self.bin_dir = bin_dir
        self.__reload_bin_files()

        self.seq_len = seq_len
        self.batch_size = batch_size

        self.current_file_index = 0
        self._change_loader()

    def _reload_bin_files(self):
        self.bin_files = list(self.bin_dir.glob("*.bin"))
        random.shuffle(self.bin_files)
        print("[Info] [pre_dataloader_manager] load", len(self.bin_files), "pre-training data files")

    def _change_loader(self):
        current_file = self.bin_files[self.current_file_index]
        print("[Info] [pre_dataloader_manager] using", current_file)
        self.current_dataloader = pre_dataloader(current_file, self.seq_len, self.batch_size)
        self.iter_count = 0

    def __iter__(self):
        reload_countdown = 10
        while True:
            for inputs, targets in self.current_dataloader:
                if self.iter_count >= 256:
                    break
                self.iter_count += 1
                yield inputs, targets

            # auto change
            self.current_file_index += 1
            if self.current_file_index >= len(self.bin_files):
                self.current_file_index = 0
            self._change_loader()

            # auto reload
            reload_countdown -= 1
            if reload_countdown <= 0:
                self._reload_bin_files()
                reload_countdown = 10


if __name__ == "__main__":
    dl = pre_dataloader("data/text.bin", 2048, 1)

    count = 0
    for i in dl:
        count += 1
        print("inputs:", i[0].shape, "targets:", i[1].shape)
        if count >= 10:
            break