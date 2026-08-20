import json
import argparse
from pathlib import Path

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("input_file")
    ap.add_argument("output_file")

    args = ap.parse_args()

    data = Path(args.input_file)
    out = Path(args.output_file)

    count = 0
    with open(data, mode='r') as f, open(out, mode='w') as o:
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            count += 1
            j = json.loads(line)
            text = ""
            if len(j["instruction"]) != 0:
                text += j["instruction"]
            if len(j["input"]) != 0:
                text += j["input"]
            if len(j["output"]) != 0:
                text += j["output"]
            if len(text) != 0:
                o.write(text + "\n")
            if count % 10000 == 0:
                print("[Concat]", count, "done")