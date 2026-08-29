import torch
import torch.nn.functional as F
from pathlib import Path
from gpt import gpt
from sft_dataloader import sft_dataloader
from tokenizer import tokenizer
from pre_training import scheduler
from lib.gpt_util import format_token
from lib.model_config import model_config
from lib.device import get_device, empty_cache, torch_amp_available

def main():
    import argparse
    ap = argparse.ArgumentParser("Quetzalcoatlus GPT-2 SFT training")
    ap.add_argument("--checkpoint", default=None,
                    help="pre-trained/resume checkpoint file path")
    args = ap.parse_args()

    if args.checkpoint is None:
        print(f"[Error] checkpoint not found, pre-training required")
        exit(1)

    tok = tokenizer(Path("data/tokenizer.json"))
    vocab_size = tok.vocab_size()
    print(f"[Info] [Tokenizer] vocab size: {vocab_size}")

    config = model_config()
    batch_size = 1

    max_steps = 2000 + 1
    warmup_steps = max_steps // 10

    grad_clip = 1.0
    grad_accum_steps = 8

    device, device_name = get_device()
    amp_enabled = torch_amp_available(device_name)

    model = gpt(vocab_size, config.d_model, config.head, config.n_layers).to(device)
    print("[Info] Model created")

    data_dir = Path("data") / "sft_training_data"
    if not data_dir.exists():
        print("[Warning] No data bin directory found:", data_dir)
        data_dir.mkdir()

    dls = [sft_dataloader(
        f, tok,
        seq_len=config.max_seq_len, batch_size=batch_size,
    ) for f in data_dir.glob("*.jsonl")]
    dls_iters = [iter(dl) for dl in dls]
    if len(dls) == 0:
        print("[Error] No jsonl found")
        exit(1)
    print("[Info] SFT jsonl:", len(dls), "files loaded")

    scaler = torch.amp.GradScaler(device_name) if amp_enabled else None
    print("[Info] Scaler ready")

    peak_lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95))
    print("[Info] Optimizer created")
    sched = scheduler(optimizer, warmup_steps, max_steps, peak_lr)
    print("[Info] Scheduler ready")

    # default start step
    start_step = 0
    # token seen, to print logs
    trained_token = 0

    ckpt_path = args.checkpoint
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        start_step = ckpt.get('SFT_step', 0)
        trained_token = ckpt.get('token_seen', 0)
        # release memory
        del ckpt
        empty_cache(device)
        print(f"[Info] loaded {ckpt_path} (step {start_step})")
    else:
        print(f"[Error] checkpoint not found: {ckpt_path}, pre-training required")
        exit(1)

    for step in range(start_step, max_steps):
        sched.step(step)

        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(grad_accum_steps):
            dl_iter = dls_iters[step % len(dls_iters)]
            inputs, targets = next(dl_iter)
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.amp.autocast(device_name, enabled=amp_enabled):
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    targets.view(-1),
                    ignore_index=-100,
                )
                loss = loss / grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()

        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        trained_token += (targets != -100).sum().item()
        print(f"[Info] step {step:5d} | loss {accum_loss:7.5f} | "
              f"lr {sched.lr:.2e} | token {format_token(trained_token)}")

        if step % 100 == 0 and step - start_step > 0:
            ckpt = {
                'SFT_step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler is not None else None,
                'token_seen': trained_token
            }
            torch.save(ckpt, f"data/sft_training_checkpoint/sft_checkpoint_step_{step}.pt")
            print(f"[Info] [checkpoint] saved at step {step}")
            if accum_loss < 0.000001:
                break


if __name__ == "__main__":
    main()
