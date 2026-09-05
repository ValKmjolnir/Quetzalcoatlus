import zipfile
from pathlib import Path

def get_checkpoint_list(root: Path) -> list[Path]:
    checkpoint_list = []
    for path in root.rglob("*.pt"):
        checkpoint_list.append(path)
    return checkpoint_list

def get_latest_checkpoint() -> Path:
    root = Path("data")
    checkpoint_list = get_checkpoint_list(root / "pre_training_checkpoint")
    checkpoint_list += get_checkpoint_list(root / "sft_training_checkpoint")

    checkpoint_list.sort(key=lambda x: x.stat().st_mtime)
    return checkpoint_list[-1]

def get_tokenizer_path() -> tuple[Path, Path]:
    root = Path("data")
    tokenizer_path = root / "tokenizer.json"
    tokenizer_log = root / "tokenizer.log"
    return tokenizer_path, tokenizer_log

def main():
    pt = get_latest_checkpoint()
    tokenizer_path, tokenizer_log = get_tokenizer_path()
    print("[Info] checkpoint path:", pt)
    print("[Info] tokenizer path:", tokenizer_path)
    print("[Info] tokenizer log path:", tokenizer_log)

    with zipfile.ZipFile("model_data.zip", "w") as zip_file:
        zip_file.write(pt, "checkpoint.pt")
        zip_file.write(tokenizer_path, "tokenizer.json")
        zip_file.write(tokenizer_log, "tokenizer.log")
    print("[Info] done")

if __name__ == "__main__":
    main()