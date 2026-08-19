from random_choose import sample_by_ratio

if __name__ == '__main__':
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("input_file")
    ap.add_argument("output_dir")
    args = ap.parse_args()

    for i in range(4):
        print(f"[Info] [RandomChoose] freq {i}")
        output = Path(args.output_dir) / f"freq_{i}.txt"
        if output.exists():
            continue
        sample_by_ratio(args.input_file, output, 0.25 / 2048)