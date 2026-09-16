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

MAGIC = b"QTOK"
CONVERT_TABLE = convert_table()

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

def convert_json(input: str, output: str):
    with open(input, "r") as f:
        data = json.load(f)


    with open(output, "wb") as f:
        # magic header
        f.write(MAGIC)

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

def main():
    ap = argparse.ArgumentParser(description="export a .json tokenizer file to C++-readable binary")
    ap.add_argument("tokenizer", help="path to a .json tokenizer file")
    ap.add_argument("-o", "--output", default="tokenizer.bin", help="output binary path")
    args = ap.parse_args()

    convert_json(args.tokenizer, args.output)

if __name__ == "__main__":
    main()
