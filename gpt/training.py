import torch
from gpt import gpt
from dataloader import data_loader

def get_lr(step: int, warmup_steps: int, max_steps: int, peak_lr: float) -> float:
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return peak_lr * max(0.0, 1.0 - progress)

def main():
    import json
    vocab_size = len(json.load(open("tokenizer.json"))["model"]["vocab"])
    print(f"actual vocab size: {vocab_size}")

    d_model = 704 // 2
    head = 11
    n_layers = 30
    batch_size = 1

    max_seq_len = 2048 // 2

    max_steps = 1000
    warmup_steps = 100
    peak_lr = 3e-4
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95))
    print("[Info] optimizer created")

    dl = data_loader("data/text.bin", seq_len=max_seq_len, batch_size=batch_size)
    dl_iter = iter(dl)
    print("[Info] data loaded")

    scaler = torch.amp.GradScaler('cuda') if cuda_available else None
    print("[Info] scaler ready")

    for step in range(max_steps):
        lr = get_lr(step, warmup_steps, max_steps, peak_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(grad_accum_steps):
            inputs, targets = next(dl_iter)
            inputs = inputs.to(device)
            targets = targets.to(device)

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

        print(f"[Info] step {step:5d} | loss {loss.item():7.4f} | lr {lr:.2e}")

        if step % 100 == 0 and step > 0:
            ckpt = {
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler is not None else None
            }
            torch.save(ckpt, f"checkpoint_step_{step}.pt")

            print(f"[Info] [checkpoint] saved at step {step}: checkpoint_step_{step}.pt")

if __name__ == "__main__":
    main()
