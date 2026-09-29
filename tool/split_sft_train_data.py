import random
import json

def split_file(input_file, output_dir):
    random.seed(114514)
    system_prompt = [
        # 主 persona（权重高）
        "你是一个聊天助手，请自然、友好地和用户聊天。",
        "你是用户专属的个人助手，请了解并尽力满足用户的需求。",
        # 变体（中等权重）
        "你是用户的聊天伙伴，回复要轻松、口语化，像朋友一样。",
        "你是我的个人助手，请用亲切、真诚的口吻回答。",
        # 合并的附加约束（低权重）
        "你是一个乐于助人的聊天助手，请用简洁、通俗、有条理的语言回答。",
        "你是用户贴心的个人助手，回答请先给出结论，需要时再分点展开。",
    ]
    weights = [6, 6, 3, 3, 1, 1]
    with open(input_file, 'r') as fin:
        count = 0
        size = 0
        file_count = 1
        output_file = f"{output_dir}/sft_freq_{file_count}.jsonl"
        fout = open(output_file, 'w')
        for line in fin:
            count += 1
            size += len(line.encode('utf-8'))
            line_json = json.loads(line)

            if len(line_json["input"]) != 0:
                continue

            data = {
                "messages": [
                    {"role": "system", "content": random.choices(system_prompt, weights=weights)[0]},
                    {"role": "user", "content": line_json["instruction"]},
                    {"role": "assistant", "content": line_json["output"]}
                ]
            }
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

            if count % 1000 == 0:
                print(f"[Info] [split_sft_train] {output_file} saved")
                fout.close()
                file_count += 1
                output_file = f"{output_dir}/sft_freq_{file_count}.jsonl"
                fout = open(output_file, 'w')
        print(f"[Info] [split_sft_train] {output_file} saved")
        fout.close()
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
