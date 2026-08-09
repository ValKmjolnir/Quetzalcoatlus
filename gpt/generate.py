import torch
from pathlib import Path
from gpt import gpt
from tokenizer import tokenizer


def generate(model, tok, prompt: str, max_new_tokens: int = 200,
             temperature: float = 0.8, top_k: int = 50, device=None):
    model.eval()

    ids = tok.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # crop to max_seq_len if needed (take last N tokens)
            crop = input_ids[:, -1024:]

            logits = model(crop)                         # (1, seq_len, vocab_size)
            logits = logits[:, -1, :]                    # (1, vocab_size) — last position only

            # temperature
            logits = logits / temperature

            # top-k filter
            if top_k > 0:
                v, _ = torch.topk(logits, top_k, dim=-1)
                thresh = v[:, -1:]                       # k-th largest value
                logits[logits < thresh] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

            input_ids = torch.cat([input_ids, next_id], dim=-1)

    full_ids = input_ids[0].tolist()
    return tok.decode(full_ids)


def main():
    import sys

    checkpoint_path = "checkpoint_step_1000.pt"
    tokenizer_path = "tokenizer.json"

    if not Path(checkpoint_path).exists():
        print(f"[Error] checkpoint not found: {checkpoint_path}")
        print("  train first, or pass a path: python gpt/generate.py checkpoint_step_XXX.pt")
        sys.exit(1)

    # load tokenizer
    tok = tokenizer(Path(tokenizer_path))

    # vocab size from tokenizer
    import json
    vocab = json.loads(Path(tokenizer_path).read_text())["model"]["vocab"]
    vocab_size = len(vocab)

    # model config — must match training
    d_model = 704 // 2
    head = 11
    n_layers = 30

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = gpt(vocab_size, d_model, head, n_layers).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    print(f"[Info] loaded {checkpoint_path} (step {ckpt['step']}) on {device}")
    print(f"[Info] type 'quit' to exit, type 'clear' to start over\n")

    memory = ""  # accumulate conversation

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if prompt.lower() == "quit":
            break
        if prompt.lower() == "clear":
            memory = ""
            print("[cleared]")
            continue

        # feed prompt + accumulated context
        full_prompt = (memory + prompt) if memory else prompt
        output = generate(model, tok, full_prompt,
                          max_new_tokens=200, temperature=0.8, top_k=50,
                          device=device)

        # strip the prompt part, show only what the model added
        generated = output[len(tok.decode(tok.encode(full_prompt))):]
        print(generated)
        print()

        # keep last ~500 chars as context for next turn
        memory = output[-500:]


if __name__ == "__main__":
    main()
