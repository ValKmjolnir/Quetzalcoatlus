import torch
import torch.nn.functional as F
from gpt import gpt
from dataloader import dataloader
from pathlib import Path
from lib.gpt_util import format_token
from lib.model_config import model_config
from lib.device import get_device, empty_cache, torch_amp_available

class scheduler:
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, peak_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.peak_lr = peak_lr
        self.lr = 0.0

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_lr * step / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        return self.peak_lr * max(0.0, 1.0 - progress)

    def step(self, step: int):
        lr = self.get_lr(step)
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def get_vocab_size(tok_json: Path) -> int:
    import json
    return len(json.load(open(tok_json))["model"]["vocab"])

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="checkpoint path, if need to resume training")
    args = ap.parse_args()

    vocab_size = get_vocab_size(Path("data/tokenizer.json"))
    print(f"[Info] Actual vocab size: {vocab_size}")

    config = model_config()
    batch_size = 1

    max_steps = 160000 + 1
    warmup_steps = max_steps // 10

    grad_clip = 1.0
    grad_accum_steps = 8

    device, device_name = get_device()
    amp_enabled = torch_amp_available(device_name)

    model = gpt(vocab_size, config.d_model, config.head, config.n_layers).to(device)
    print("[Info] Model created")

    bin_dir = Path("data")
    if not bin_dir.exists():
        print("[Warning] No data bin directory found:", bin_dir)
        bin_dir.mkdir()

    dls = [dataloader(str(f), seq_len=config.max_seq_len, batch_size=batch_size) for f in bin_dir.glob("*.bin")]
    dls_iters = [iter(dl) for dl in dls]
    if len(dls) == 0:
        print("[Error] No data bin found")
        exit(1)
    print("[Info] Data bins:", len(dls), "files loaded")

    scaler = torch.amp.GradScaler(device_name) if amp_enabled else None
    print("[Info] scaler ready")

    peak_lr = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95))
    print("[Info] Optimizer created")
    sched = scheduler(optimizer, warmup_steps, max_steps, peak_lr)
    print("[Info] Scheduler ready")

    # default start step
    start_step = 0
    # token seen, to print logs
    token_seen = 0

    # load checkpoint if exists
    if args.checkpoint is not None and Path(args.checkpoint).exists():
        checkpoint = Path(args.checkpoint)
        print("[Info] Resuming training from checkpoint:", checkpoint)
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        start_step = ckpt['step'] + 1
        token_seen = ckpt.get('token_seen', 0)
        del ckpt
        empty_cache(device)
    else:
        print("[Info] Starting new training")

    for step in range(start_step, max_steps):
        sched.step(step)

        optimizer.zero_grad()
        accum_loss = 0.0

        # accumulate micro-batch
        for _ in range(grad_accum_steps):
            dl_iter = dls_iters[step % len(dls_iters)]
            inputs, targets = next(dl_iter)
            inputs = inputs.to(device)
            targets = targets.to(device)

            token_seen += inputs.numel()

            with torch.amp.autocast(device_name, enabled=amp_enabled):
                logits = model(inputs)            # (batch, seq_len, vocab_size)
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),  # (batch * seq_len, vocab_size)
                    targets.view(-1)              # (batch * seq_len)
                )
                loss = loss / grad_accum_steps    # normalizing

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()

        # acc all micro-batch
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        print(f"[Info] step {step:5d} | loss {accum_loss:7.5f} | "
              f"lr {sched.lr:.2e} | token {format_token(token_seen)}")

        if step % 100 == 0 and step > 0:
            ckpt = {
                'step': step,
                'SFT_step': 0,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler is not None else None,
                'token_seen': token_seen
            }
            torch.save(ckpt, f"data/checkpoint_step_{step}.pt")

            print(f"[Info] [checkpoint] saved at step {step}: data/checkpoint_step_{step}.pt")

if __name__ == "__main__":
    main()
