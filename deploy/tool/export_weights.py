import argparse
import struct

MAGIC = b"QGPT"
DTYPE_F32 = 0 # mark tensor is f32 based

def write_tensor(f, name: str, shape, data: bytes):
    # [name_len][name_bytes][ndim][dim0][dim1]...[dimN]...[dtype][raw_fp_data]

    name_b = name.encode("utf-8")
    # length of name; name bytes;
    f.write(struct.pack("<I", len(name_b)))
    f.write(name_b)

    # length of shape; shape;
    f.write(struct.pack("<I", len(shape)))
    for s in shape:
        f.write(struct.pack("<I", int(s)))

    # dtype; data
    f.write(struct.pack("<I", DTYPE_F32))
    f.write(data)

def convert_weights(checkpoint: str, output: str):
    import torch

    ckpt = torch.load(checkpoint,
                      map_location="cpu", # load into cpu memory
                      weights_only=False) # allow loading the dict object
    state = ckpt.get("model", ckpt)       # training ckpt dict, or a bare state_dict

    seen = {}  # data_ptr -> canonical name, for weight tying
    count = 0
    total_params = 0
    with open(output, "wb") as f:
        # magic header
        f.write(MAGIC)

        # here to store the count of tensors, but now we don't know it
        # so we write a placeholder here
        count_pos = f.tell()
        f.write(struct.pack("<I", 0))

        for name, t in state.items():
            ptr = t.data_ptr()
            if ptr in seen:
                print(f"[skip] {name}  (tied to {seen[ptr]})")
                continue
            seen[ptr] = name

            # detach from the calculation graph, make sure it's on CPU and contiguous
            t = t.detach().cpu().contiguous().float()
            shape = tuple(t.shape)
            write_tensor(f, name, shape, t.numpy().tobytes())
            count += 1
            total_params += t.numel()
            print(f"[write] {name} {shape}")

        # write the count of tensors back
        end = f.tell()
        f.seek(count_pos)
        f.write(struct.pack("<I", count))
        f.seek(end)

    print(f"[done] {count} tensors, {total_params} params -> {output}")

def main():
    ap = argparse.ArgumentParser(description="export a .pt checkpoint to C++-readable binary")
    ap.add_argument("checkpoint", help="path to a .pt checkpoint (torch.save output)")
    ap.add_argument("-o", "--output", default="weights.bin", help="output binary path")
    args = ap.parse_args()

    convert_weights(args.checkpoint, args.output)

if __name__ == "__main__":
    main()
