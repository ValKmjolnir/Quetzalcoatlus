import random

def sample_by_ratio(input_file, output_file, ratio=0.25 / 2048):
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        count = 0
        size = 0
        for line in fin:
            if random.random() < ratio:
                count += 1
                size += len(line.encode('utf-8'))
                fout.write(line)

                if count % 1000 == 0:
                    print(f"{count} lines, {size / 1024 / 1024:.2f} MB")
        print(f"{count} lines, {size / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("input_file", help="input large text file (really large!)")
    ap.add_argument("output_dir", help="output directory")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.exists():
        out_dir.mkdir()

    # random choose 4 times each time the script run
    count = 0
    i = 0
    while True:
        i += 1
        output = out_dir / f"freq_{i}.txt"
        print(f"[Info] [RandomChoose] choosing {i} to {output}")
        if output.exists():
            print(f"[Info] [RandomChoose] {output} exists, skip")
            continue
        count += 1
        sample_by_ratio(args.input_file, output, 0.25 / 2048)
        if count >= 4:
            break