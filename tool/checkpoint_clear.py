from pathlib import Path

def get_checkpoint_list(checkpoint_dir: Path) -> list[Path]:
    res = []
    for f in checkpoint_dir.rglob("*.pt"):
        res.append(f)

    res.sort(key = lambda x: x.stat().st_mtime)
    return res

def main():
    pre_train_path = Path("data") / "pre_training_checkpoint"
    sft_train_path = Path("data") / "sft_training_checkpoint"

    reserved = []

    pre_train_checkpoints = get_checkpoint_list(pre_train_path)
    sft_train_checkpoints = get_checkpoint_list(sft_train_path)
    if len(pre_train_checkpoints) > 4:
        reserved.extend(pre_train_checkpoints[-4:])
    else:
        reserved.extend(pre_train_checkpoints)
    if len(sft_train_checkpoints) > 4:
        reserved.extend(sft_train_checkpoints[-4:])
    else:
        reserved.extend(sft_train_checkpoints)

    for i in pre_train_checkpoints + sft_train_checkpoints:
        if i not in reserved:
            print(f"delete: {i}")
            i.unlink()
    for i in reserved:
        print(f"reserved: {i}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="watch mode")
    args = parser.parse_args()

    if args.watch:
        import time
        import sys

        try:
            while True:
                print("[Info] watching...")
                time.sleep(60 * 4)
                main()
        except KeyboardInterrupt:
            sys.exit(0)
    else:
        main()