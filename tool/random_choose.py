import random

def sample_by_ratio(input_file, output_file, ratio=0.25 / 8):
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

    ap = argparse.ArgumentParser()
    ap.add_argument("input_file")
    ap.add_argument("output_file")
    ap.add_argument("--ratio", type=float, default=0.25 / 1024)

    args = ap.parse_args()

    sample_by_ratio(args.input_file, args.output_file, args.ratio)