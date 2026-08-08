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

    def encode(self, input: str) -> list[int]:
        unicode_chars = [self.byte_to_unicode[i] for i in input.encode("utf-8")]
        ids = [self.vocab[ch] for ch in unicode_chars]

        for lhs, rhs in self.merges:
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

if __name__ == "__main__":
    tok = tokenizer(Path("debug.json"))

    ids = tok.encode("大家好啊，我是电棍，今天给大家看点想看的东西")
    raw = tok.decode(ids)
    print("ids:", ids)
    print("raw:", raw)

    ids = tok.encode("啊↑啊↓，啊默写啊我写诗，啊莫一波三把跪，啊米灭两坨fish，一把米吧别谢，啊米浴说的道↑理↓→5")
    raw = tok.decode(ids)
    print("ids:", ids)
    print("raw:", raw)
