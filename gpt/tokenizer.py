import json
from pathlib import Path


class tokenizer:
    def __init__(self, vocab_file: Path):
        if not vocab_file.exists():
            raise FileNotFoundError(f"{vocab_file} not found")
        data = json.loads(vocab_file.read_text())

        self.added_tokens = data.get("added_tokens") or []
        self.vocab = data["model"].get("vocab") or {}
        self.id_to_str = {v: k for k, v in self.vocab.items()}
        self.merges = data["model"].get("merges") or []

        self._generate_convert_table()

    def _generate_convert_table(self):
        self.byte_to_unicode = {}
        n = 0
        for i in range(256):
            visible = (33 <= i <= 126) or (161 <= i <= 172) or (174 <= i <= 255)
            if visible:
                self.byte_to_unicode[i] = chr(i)
            else:
                self.byte_to_unicode[i] = chr(256 + n)
                n += 1

    def _init_index_list(self, source: bytes) -> list[int]:
        special_tokens = [info["content"].encode("utf-8") for info in self.added_tokens]
        special_tokens.sort(key=len, reverse=True)
        index = 0
        
        ids = []
        while index < len(source):
            if chr(source[index]) == '<':
                find = False
                for tok in special_tokens:
                    if source.find(tok, index) == index:
                        ids.append(self.vocab[tok.decode()])
                        index += len(tok)
                        find = True
                        break
                if find:
                    continue
            unicode_char = self.byte_to_unicode[source[index]]
            ids.append(self.vocab[unicode_char])
            index += 1
        return ids

    def encode(self, input: str, show_process: bool=False) -> list[int]:
        source = input.encode("utf-8", "ignore")
        ids = self._init_index_list(source)

        if show_process:
            from tqdm import tqdm
            merge_iter = tqdm(self.merges, desc="Encoding")
        else:
            merge_iter = self.merges

        for lhs, rhs in merge_iter:
            l = self.vocab[lhs]
            r = self.vocab[rhs]
            merged_str = lhs + rhs
            merged_id = self.vocab[merged_str]

            new_ids = []
            i = 0
            while i < len(ids):
                if (i + 1 < len(ids) and ids[i] == l and ids[i + 1] == r):
                    new_ids.append(merged_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids

        return ids

    def decode(self, input: list[int]) -> str:
        unicode_str = ''.join(self.id_to_str[i] for i in input)
        unicode_to_byte = {v: k for k, v in self.byte_to_unicode.items()}
        bytes_ = bytes(unicode_to_byte[ch] for ch in unicode_str)

        return bytes_.decode("utf-8", errors="replace")

    def vocab_size(self) -> int:
        return len(self.vocab)

def text_to_bin(tok_json: Path, input: Path, output: Path):
    import numpy as np

    text = open(input).read()
    tok = tokenizer(tok_json)
    ids = tok.encode(text, show_process=True)

    np.array(ids, dtype=np.uint32).tofile(output)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Quetzalcoatlus GPT-2 tokenizer")
    ap.add_argument("--prepare", action="store_true", help="prepare pretrain data")
    args = ap.parse_args()

    tok = tokenizer(Path("tokenizer.json"))

    if not args.prepare:
        print("====================== TEST ======================")

        sentences = [
            "大家好啊，我是电棍，今天给大家看点想看的东西",
            "啊↑啊↓，啊默写啊我写诗，啊莫一波三把跪，啊米灭两坨fish，一把米吧别谢，啊米浴说的道↑理↓→",
            "5 年，你知道我这 5 年都是怎么过的吗",
            "我去不早说",
            "事情终于有了新的退展",
            "<|im_start|><|im_end|><|pad|><|unk|><|endoftext|>",
            "<|im_start|>你好<|im_end|>hello？<|pad|>"
        ]

        for s in sentences:
            ids = tok.encode(s)
            raw = tok.decode(ids)
            print("ids:", ids)
            print("raw:", raw)

    if args.prepare:
        print("====================== CONV ======================")
        text_to_bin(
            Path("tokenizer.json"),
            Path("data/text.txt"),
            Path("data/text.bin"))
        print("====================== CONV[DONE] ================")
