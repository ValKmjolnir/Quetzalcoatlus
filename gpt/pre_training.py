import torch
from gpt import gpt
from dataloader import dataloader
from pathlib import Path

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

def format_token(token_size: int) -> str:
    if token_size < 1e3:
        return f"{token_size}"
    elif token_size < 1e6:
        return f"{token_size / 1e3:.2f}k"
    elif token_size < 1e9:
        return f"{token_size / 1e6:.2f}M"
    return f"{token_size / 1e9:.2f}B"
def main():
    import json
    vocab_size = len(json.load(open("data/tokenizer.json"))["model"]["vocab"])
    print(f"[Info] Actual vocab size: {vocab_size}")

    d_model = 704 // 2
    head = 11
    n_layers = 30
    batch_size = 1

    max_seq_len = 2048 // 2

    max_steps = 8000 + 1
    warmup_steps = max_steps // 10

    grad_clip = 1.0
    grad_accum_steps = 8

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("[Warning] CUDA device not available")
    else:
        print("[Info] CUDA device available")

    device = torch.device('cuda' if cuda_available else 'cpu')
    model = gpt(vocab_size, d_model, head, n_layers).to(device)
    print("[Info] Model created")

    bin_dir = Path("data")
    dls = [dataloader(str(f), seq_len=max_seq_len, batch_size=batch_size) for f in bin_dir.glob("*.bin")]
    dls_iters = [iter(dl) for dl in dls]
    print("[Info] Data loaded")

    scaler = torch.amp.GradScaler('cuda') if cuda_available else None
    print("[Info] scaler ready")

    peak_lr = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95))
    print("[Info] Optimizer created")
    sched = scheduler(optimizer, warmup_steps, max_steps, peak_lr)
    print("[Info] Scheduler ready")

    token_seen = 0
    for step in range(max_steps):
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

            with torch.amp.autocast('cuda', enabled=cuda_available):
                logits = model(inputs)            # (batch, seq_len, vocab_size)
                loss = torch.nn.functional.cross_entropy(
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
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler is not None else None
            }
            torch.save(ckpt, f"data/checkpoint_step_{step}.pt")

            print(f"[Info] [checkpoint] saved at step {step}: data/checkpoint_step_{step}.pt")

if __name__ == "__main__":
    main()
