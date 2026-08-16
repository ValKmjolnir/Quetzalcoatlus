import torch
import torch.nn.functional as F
from pathlib import Path
from gpt import gpt
from sft_dataloader import sft_dataloader
from tokenizer import tokenizer
from pre_training import scheduler


def main():
    import argparse
    ap = argparse.ArgumentParser("Quetzalcoatlus GPT-2 SFT training")
    ap.add_argument("checkpoint", nargs="?", default="data/checkpoint_step_1500.pt",
                    help="pre-trained checkpoint file path")
    args = ap.parse_args()

    tok = tokenizer(Path("data/tokenizer.json"))
    vocab_size = tok.vocab_size()
    print(f"[Info] [Tokenizer] vocab size: {vocab_size}")

    d_model = 704 // 2
    head = 11
    n_layers = 30
    batch_size = 1

    seq_len = 1024

    max_steps = 1000 + 1
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

    ckpt_path = args.checkpoint
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        step = ckpt['step']
        # release memory
        del ckpt
        if cuda_available:
            torch.cuda.empty_cache()
        print(f"[Info] loaded {ckpt_path} (step {step})")
    else:
        print(f"[Warning] checkpoint not found: {ckpt_path}, training from scratch")

    print("[Info] Model created")

    dl = sft_dataloader(
        Path("data/SFT.jsonl"), tok,
        seq_len=seq_len, batch_size=batch_size,
    )
    dl_iter = iter(dl)
    print("[Info] SFT data loaded")

    scaler = torch.amp.GradScaler('cuda') if cuda_available else None
    print("[Info] scaler ready")

    peak_lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95))
    print("[Info] Optimizer created")
    sched = scheduler(optimizer, warmup_steps, max_steps, peak_lr)
    print("[Info] Scheduler ready")

    trained_token = 0
    for step in range(max_steps):
        sched.step(step)

        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(grad_accum_steps):
            inputs, targets = next(dl_iter)
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.amp.autocast('cuda', enabled=cuda_available):
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
              f"lr {sched.lr:.2e} | train_tok {trained_token / 1e6:.2f}M")

        if step % 50 == 0 and step > 0:
            ckpt = {
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler is not None else None,
            }
            torch.save(ckpt, f"data/sft_checkpoint_step_{step}.pt")
            print(f"[Info] [checkpoint] saved at step {step}")
            if accum_loss < 0.000001:
                break


if __name__ == "__main__":
    main()
