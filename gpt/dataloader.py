import numpy as np
import torch

class dataloader:
    def __init__(self, bin_path, seq_len: int, batch_size: int):
        self.data = np.memmap(bin_path, dtype=np.uint32, mode='r')
        self.seq_len = seq_len
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            starts = np.random.randint(
                0, len(self.data) - self.seq_len - 1,
                size=self.batch_size)
            inputs = np.stack([self.data[s : s + self.seq_len] for s in starts])
            targets = np.stack([self.data[s + 1 : s + self.seq_len + 1] for s in starts])

            yield torch.from_numpy(inputs).long(), torch.from_numpy(targets).long()

if __name__ == "__main__":
    dl = dataloader("data/text.bin", 2048, 1)

    count = 0
    for i in dl:
        count += 1
        print("inputs:", i[0].shape, "targets:", i[1].shape)
        if count >= 10:
            break