from pathlib import Path

data_dir = Path("data")
if not data_dir.exists():
    print("[Info] Data directory created")
    data_dir.mkdir()

pre_training_data_dir = data_dir / "pre_training_data"
if not pre_training_data_dir.exists():
    print("[Info] Pre-training data directory created")
    pre_training_data_dir.mkdir()
pre_training_checkpoint_dir = data_dir / "pre_training_checkpoint"
if not pre_training_checkpoint_dir.exists():
    print("[Info] Pre-training checkpoint directory created")
    pre_training_checkpoint_dir.mkdir()

sft_training_data_dir = data_dir / "sft_training_data"
if not sft_training_data_dir.exists():
    print("[Info] SFT data directory created")
    sft_training_data_dir.mkdir()
sft_training_checkpoint_dir = data_dir / "sft_training_checkpoint"
if not sft_training_checkpoint_dir.exists():
    print("[Info] SFT checkpoint directory created")
    sft_training_checkpoint_dir.mkdir()

print("[Info] Data directories ready")