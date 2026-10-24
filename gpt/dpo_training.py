import torch
import torch.nn.functional as F
import time

from pathlib import Path
from gpt import gpt
from dpo_dataloader import dpo_dataloader_manager
from tokenizer import tokenizer
from pre_training import scheduler
from lib.gpt_util import format_token, logtime
from lib.model_config import model_config
from lib.device import get_device, empty_cache, torch_amp_available
from lib.checkpoint_clear import checkpoint_clear

def seq_logp(logits: torch.Tensor, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # logits (B, L, V), seq (B, L), mask (B, L) bool: 1 on completion tokens
    # sum of per-token log-probs over the completion span only
    logp = F.log_softmax(logits, dim=-1)                                  # (B, L, V)
    tok = logp[:, :-1].gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)   # (B, L-1)
    return (tok * mask[:, 1:].float()).sum(-1)                            # (B,)


def main():
    import argparse
    ap = argparse.ArgumentParser("Quetzalcoatlus GPT-2 DPO training")
    ap.add_argument("--checkpoint", default=None,
                    help="policy checkpoint (SFT ckpt to start, DPO ckpt to resume)")
    ap.add_argument("--ref", default=None,
                    help="reference checkpoint (SFT ckpt, kept frozen)")
    ap.add_argument("--max_steps", type=int, default=3000, help="number of optimizer steps")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    ap.add_argument("--silent", action="store_true", default=False, help="silent mode")
    args = ap.parse_args()

    if args.checkpoint is None or args.ref is None:
        print("[Error] --checkpoint and --ref are required (both point to the SFT ckpt when starting DPO)")
        exit(1)

    tok = tokenizer(Path("data/tokenizer.json"))
    vocab_size = tok.vocab_size()
    print(f"[Info] [Tokenizer] vocab size: {vocab_size}")

    config = model_config()
    batch_size = 1

    max_steps = args.max_steps
    warmup_steps = max_steps // 10
    beta = args.beta

    grad_clip = 1.0
    grad_accum_steps = 16

    device, device_name = get_device()
    amp_enabled = torch_amp_available(device_name)

    model = gpt(vocab_size, config.d_model, config.head, config.n_layers).to(device)
    ref_model = gpt(vocab_size, config.d_model, config.head, config.n_layers).to(device)
    print("[Info] Model created")

    data_dir = Path("data") / "rlhf_training_data"
    if not data_dir.exists():
        data_dir.mkdir()
    if len(list(data_dir.glob("*.npz"))) == 0:
        print("[Error] no npz files found, run `python gpt/dpo_dataloader.py --prepare` first")
        exit(1)

    dl = iter(dpo_dataloader_manager(data_dir, tok, seq_len=config.max_seq_len, batch_size=batch_size))

    scaler = torch.amp.GradScaler(device_name) if amp_enabled else None
    print("[Info] Scaler ready")

    peak_lr = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95))
    print("[Info] Optimizer created")
    sched = scheduler(optimizer, warmup_steps, max_steps, peak_lr)
    print("[Info] Scheduler ready")

    start_step = 0
    trained_token = 0

    ckpt_path = args.checkpoint
    if not Path(ckpt_path).exists():
        print(f"[Error] checkpoint not found: {ckpt_path}")
        exit(1)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    if ckpt.get('DPO_step', 0) != 0:
        optimizer.load_state_dict(ckpt['optimizer'])
        if scaler is not None and ckpt.get('scaler') is not None:
            scaler.load_state_dict(ckpt['scaler'])
        start_step = ckpt['DPO_step'] + 1
        trained_token = ckpt.get('token_seen', 0)
        print(f"[Info] resumed DPO from {ckpt_path} (step {start_step})")
    else:
        print(f"[Info] starting DPO from SFT checkpoint {ckpt_path}")
    del ckpt
    empty_cache(device)

    ref_path = args.ref
    if not Path(ref_path).exists():
        print(f"[Error] reference checkpoint not found: {ref_path}")
        exit(1)
    ref_ckpt = torch.load(ref_path, map_location=device, weights_only=False)
    ref_model.load_state_dict(ref_ckpt['model'])
    del ref_ckpt
    ref_model.requires_grad_(False)
    ref_model.eval()
    empty_cache(device)
    print(f"[Info] loaded frozen reference model from {ref_path}")

    for step in range(start_step, max_steps):
        sched.step(step)

        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(grad_accum_steps):
            chosen_ids, chosen_mask, rejected_ids, rejected_mask = next(dl)
            chosen_ids = chosen_ids.to(device)
            chosen_mask = chosen_mask.to(device)
            rejected_ids = rejected_ids.to(device)
            rejected_mask = rejected_mask.to(device)

            trained_token += (chosen_mask.sum() + rejected_mask.sum()).item()

            with torch.amp.autocast(device_name, enabled=amp_enabled):
                logits_c = model(chosen_ids)
                logits_r = model(rejected_ids)
                with torch.no_grad():
                    ref_logits_c = ref_model(chosen_ids)
                    ref_logits_r = ref_model(rejected_ids)

            # DPO loss is precision-sensitive: keep logp math in fp32
            logp_c = seq_logp(logits_c.float(), chosen_ids, chosen_mask)
            logp_r = seq_logp(logits_r.float(), rejected_ids, rejected_mask)
            ref_c = seq_logp(ref_logits_c.float(), chosen_ids, chosen_mask)
            ref_r = seq_logp(ref_logits_r.float(), rejected_ids, rejected_mask)

            loss = -F.logsigmoid(beta * ((logp_c - ref_c) - (logp_r - ref_r)))
            loss = loss.mean() / grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()

        if args.silent:
            time.sleep(0.5)

        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        print(f"[Info] {logtime()} step {step:5d} | loss {accum_loss:7.5f} | "
              f"lr {sched.lr:.2e} | token {format_token(trained_token)}")

        if step % 100 == 0 and step > 0:
            ckpt = {
                'DPO_step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler is not None else None,
                'token_seen': trained_token,
            }
            torch.save(ckpt, f"data/rlhf_training_checkpoint/dpo_checkpoint_step_{step}.pt")
            print(f"[Info] [checkpoint] saved at step {step}")
            checkpoint_clear()
            if args.silent:
                for i in range(24):
                    print(f"[Info] sleep for 10 seconds ({i + 1}/24)")
                    time.sleep(10)


if __name__ == "__main__":
    main()
