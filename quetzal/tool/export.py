import argparse
import json
import struct

class convert_table:
    def __init__(self):
        self.byte_to_unicode = {}
        n = 0
        for i in range(256):
            visible = (33 <= i <= 126) or (161 <= i <= 172) or (174 <= i <= 255)
            if visible:
                self.byte_to_unicode[i] = chr(i)
            else:
                self.byte_to_unicode[i] = chr(256 + n)
                n += 1
        self.unicode_to_byte = {v: k for k, v in self.byte_to_unicode.items()}

    def decode(self, unicode_str: str) -> bytes:
        return bytes([self.unicode_to_byte[c] for c in unicode_str])

MAGIC = b"QGPT"
CONVERT_TABLE = convert_table()
DTYPE_F32 = 0 # mark tensor is f32 based

def write_token(f, id: int, content: str):
    f.write(struct.pack("<I", id))
    content_b = CONVERT_TABLE.decode(content)
    f.write(struct.pack("<I", len(content_b)))
    f.write(content_b)

def write_vocab(f, vocab: dict[str, int]):
    f.write(struct.pack("<I", len(vocab)))
    for k, v in vocab.items():
        write_token(f, v, k)

def write_merges(f, merges: list[tuple[str, str]]):
    f.write(struct.pack("<I", len(merges)))
    for a, b in merges:
        content_a = CONVERT_TABLE.decode(a)
        content_b = CONVERT_TABLE.decode(b)
        f.write(struct.pack("<I", len(content_a)))
        f.write(content_a)
        f.write(struct.pack("<I", len(content_b)))
        f.write(content_b)

def convert_json(input: str, output: str, f):
    with open(input, "r") as r:
        data = json.load(r)

    # write added tokens
    f.write(struct.pack("<I", len(data["added_tokens"])))
    for added_token in data["added_tokens"]:
        content = CONVERT_TABLE.decode(added_token["content"])
        f.write(struct.pack("<I", len(content)))
        f.write(content)

    # write vocab
    write_vocab(f, data["model"]["vocab"])
    # write merges
    write_merges(f, data["model"]["merges"])

    print(f"[done] {input} -> {output}")

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

def convert_weights(checkpoint: str, output: str, f):
    import torch

    print(f"[load] loading model from {checkpoint}...")
    ckpt = torch.load(checkpoint,
                      map_location="cpu", # load into cpu memory
                      weights_only=False) # allow loading the dict object
    state = ckpt.get("model", ckpt)       # training ckpt dict, or a bare state_dict

    seen = {}  # data_ptr -> canonical name, for weight tying
    count = 0
    total_params = 0

    # here to store the count of tensors, but now we don't know it
    # so we write a placeholder here
    count_pos = f.tell()
    f.write(struct.pack("<I", 0))

    for name, t in state.items():
        ptr = t.data_ptr()
        if ptr in seen:
            print(f"\033[2K[skip] {name}  (tied to {seen[ptr]}) [{count}]", end="\r")
            continue
        seen[ptr] = name

        # detach from the calculation graph, make sure it's on CPU and contiguous
        t = t.detach().cpu().contiguous().float()
        shape = tuple(t.shape)
        write_tensor(f, name, shape, t.numpy().tobytes())
        count += 1
        total_params += t.numel()
        print(f"\033[2K[write] {name} {shape} [{count}]", end="\r")

    print()
    # write the count of tensors back
    end = f.tell()
    f.seek(count_pos)
    f.write(struct.pack("<I", count))
    f.seek(end)

    print(f"[done] {count} tensors, {total_params} params -> {output}")

def write_config_string(f, s: str):
    content = s.encode("utf-8")
    f.write(struct.pack("<I", len(content)))
    f.write(content)

def convert_model_config(input: str, output: str, f):
    CONFIG_UINT32_TYPE = 0
    CONFIG_STRING_TYPE = 1
    with open(input, "r") as r:
        data = json.load(r)

    total_params = len(data)
    f.write(struct.pack("<I", total_params))
    for k, v in data.items():
        write_config_string(f, k)
        if isinstance(v, str):
            f.write(struct.pack("<I", CONFIG_STRING_TYPE))
            write_config_string(f, v)
        elif isinstance(v, int):
            f.write(struct.pack("<I", CONFIG_UINT32_TYPE))
            f.write(struct.pack("<I", v))
        else:
            print(f"[Error] unsupported type: {type(v)} {v}")

    print(f"[done] model config {total_params} params -> {output}")

def main():
    ap = argparse.ArgumentParser(description="export model to C++-readable binary")
    ap.add_argument("model_config", help="path of .json model config file")
    ap.add_argument("checkpoint", help="path of .pt checkpoint (torch.save output)")
    ap.add_argument("tokenizer", help="path pf .json tokenizer file")
    ap.add_argument("-o", "--output", default="model.bin", help="output binary path")
    args = ap.parse_args()

    with open(args.output, "wb") as f:
        # magic header
        f.write(MAGIC)
        convert_model_config(args.model_config, args.output, f)
        convert_weights(args.checkpoint, args.output, f)
        convert_json(args.tokenizer, args.output, f)

if __name__ == "__main__":
    main()
