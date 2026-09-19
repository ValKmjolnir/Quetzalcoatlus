import random
import json

def sample_by_ratio(input_file, output_file, ratio=0.25 / 2048):
    system_prompt = [
        "You are a helpful assistant.",
        "You are chatting with other people.",
        "You are enjoying the conversation.",
        "You are having a good time."
    ]
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        count = 0
        size = 0
        for line in fin:
            if random.random() < ratio:
                count += 1
                size += len(line.encode('utf-8'))
                line_json = json.loads(line)

                if len(line_json["input"]) != 0:
                    continue

                data = {
                    "messages": [
                        {"role": "system", "content": system_prompt[random.randint(0, len(system_prompt) - 1)]},
                        {"role": "user", "content": line_json["instruction"]},
                        {"role": "assistant", "content": line_json["output"]}
                    ]
                }
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")

                if count % 1000 == 0:
                    print(f"{count} lines, {size / 1024 / 1024:.2f} MB")
        print(f"{count} lines, {size / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("input_file", help="input large text file (really large!)")
    ap.add_argument("output_dir", help="output directory")
    ap.add_argument("--file_count", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.exists():
        out_dir.mkdir()

    # random choose {file_count} times each time the script run
    count = 0
    i = 0
    while True:
        i += 1
        output = out_dir / f"sft_freq_{i}.jsonl"
        if output.exists():
            print(f"[Info] [RandomChoose] {output} exists, skip")
            continue
        print(f"[Info] [RandomChoose] choosing {i} to {output}")
        count += 1
        sample_by_ratio(args.input_file, output, 0.25 / 1024)
        if count >= args.file_count:
            break
