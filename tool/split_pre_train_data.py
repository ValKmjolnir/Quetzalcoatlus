def split_file(input_file, output_dir):
    with open(input_file, 'r') as fin:
        count = 0
        size = 0
        file_count = 1
        output_file = f"{output_dir}/freq_{file_count}.txt"
        fout = open(output_file, 'w')
        for line in fin:
            count += 1
            size += len(line.encode('utf-8'))
            fout.write(line)
            if count % 1000 == 0:
                print(f"{count} lines, {size / 1024 / 1024:.2f} MB")
            if count % 10000 == 0:
                print(f"[Info] [RandomChoose] {output_file} saved")
                fout.close()
                file_count += 1
                output_file = f"{output_dir}/freq_{file_count}.txt"
                fout = open(output_file, 'w')
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

    split_file(args.input_file, args.output_dir)
