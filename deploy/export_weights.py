"""Export a PyTorch checkpoint to a plain binary that C++ can read.

torch.save writes a pickle/zipfile container that has no trivial C++ reader.
This script translates the model state_dict once, on the Python side, into a
self-describing little-endian binary layout:

    magic        4 bytes   "QGPT"
    num_tensors  uint32
    per tensor:
        name_len  uint32
        name      bytes (utf-8)
        ndim      uint32
        shape     uint32 x ndim
        dtype     uint32   (0 = float32)
        data      raw float32, row-major

Weight tying (tok_emb.weight == lm_head.weight) is de-duplicated by data_ptr,
so the tied matrix is written exactly once.

Usage:
    python export_weights.py data/checkpoint_step_1500.pt -o weights.bin
"""
import argparse
import struct

MAGIC = b"QGPT"
DTYPE_F32 = 0


def _write_tensor(f, name: str, shape, data: bytes):
    name_b = name.encode("utf-8")
    f.write(struct.pack("<I", len(name_b)))
    f.write(name_b)
    f.write(struct.pack("<I", len(shape)))
    for s in shape:
        f.write(struct.pack("<I", int(s)))
    f.write(struct.pack("<I", DTYPE_F32))
    f.write(data)


def main():
    import torch  # lazy import so --help / py_compile work without torch installed

    ap = argparse.ArgumentParser(description="export a .pt checkpoint to C++-readable binary")
    ap.add_argument("checkpoint", help="path to a .pt checkpoint (torch.save output)")
    ap.add_argument("-o", "--output", default="weights.bin", help="output binary path")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)  # training ckpt dict, or a bare state_dict

    seen = {}  # data_ptr -> canonical name, for weight tying
    count = 0
    total_params = 0
    with open(args.output, "wb") as f:
        f.write(MAGIC)
        count_pos = f.tell()
        f.write(struct.pack("<I", 0))  # patched at the end

        for name, t in state.items():
            ptr = t.data_ptr()
            if ptr in seen:
                print(f"[skip] {name}  (tied to {seen[ptr]})")
                continue
            seen[ptr] = name

            t = t.detach().cpu().contiguous().float()
            shape = tuple(t.shape)
            _write_tensor(f, name, shape, t.numpy().tobytes())
            count += 1
            total_params += t.numel()
            print(f"[write] {name} {shape}")

        end = f.tell()
        f.seek(count_pos)
        f.write(struct.pack("<I", count))
        f.seek(end)

    print(f"[done] {count} tensors, {total_params} params -> {args.output}")


if __name__ == "__main__":
    main()
