import torch
from pathlib import Path
from gpt import gpt
from tokenizer import tokenizer

def generate_chat(model: gpt, tok: tokenizer,
                  prompt: str, max_new_tokens: int = 512,
                  temperature: float = 0.8, top_k: int = 50, device=None):
    model.eval()

    ids = tok.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # crop to max_seq_len if needed (take last N tokens)
            crop = input_ids[:, -1024:]

            logits = model(crop)      # (1, seq_len, vocab_size)
            logits = logits[:, -1, :] # (1, vocab_size) — last position only

            # temperature
            logits = logits / temperature

            # top-k filter
            if top_k > 0:
                v, _ = torch.topk(logits, top_k, dim=-1)
                thresh = v[:, -1:] # k-th largest value
                logits[logits < thresh] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1) # (1, 1)
            input_ids = torch.cat([input_ids, next_id], dim=-1)

            # stop generation if we generated <|im_end|>
            if next_id.item() == tok.vocab["<|im_end|>"]:
                break

    gen_ids = input_ids[0].tolist()[prompt_len:]

    # strip trailing im_end
    im_end_id = tok.vocab["<|im_end|>"]
    if gen_ids and gen_ids[-1] == im_end_id:
        gen_ids = gen_ids[:-1]

    return tok.decode(gen_ids)


def chat_loop(model: gpt, tok: tokenizer, device, system_prompt: str):
    messages = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() == "quit" or user_input.lower() == "exit":
            print("\n[Quetzal] [Exit]\n")
            break
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": system_prompt}]
            print("\n[Quetzal] [Cleared]\n")
            continue
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        prompt = ""
        for m in messages:
            prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        response = generate_chat(model, tok, prompt,
                                 max_new_tokens=512, temperature=0.8, top_k=50,
                                 device=device)
        print("\n[Quetzal]", response)
        print()

        messages.append({"role": "assistant", "content": response})

        # keep context within ~800 tokens of content
        approx_tokens = sum(len(m["content"]) // 4 for m in messages)
        while approx_tokens > 800 and len(messages) > 2:
            messages.pop(1)  # drop oldest non-system message
            approx_tokens = sum(len(m["content"]) // 4 for m in messages)


def logo_dump():
    logo = """
      __   _  _  ____  ____  ____   __   __   
     /  \\ / )( \\(  __)(_  _)(__  ) / _\\ (  )  
    (  O )) \\/ ( ) _)   )(   / _/ /    \\/ (_/\\
     \\__\\)\\____/(____) (__) (____)\\_/\\_/\\____/
      ___  __    __  ____  __    _  _  ____   
     / __)/  \\  / _\\(_  _)(  )  / )( \\/ ___)  
    ( (__(  O )/    \\ )(  / (_/\\) \\/ (\\___ \\  
     \\___)\\__/ \\_/\\_/(__) \\____/\\____/(____/  
    """

    print(logo)

def main():
    import sys
    import argparse

    logo_dump()

    ap = argparse.ArgumentParser("Quetzalcoatlus GPT-2 text generation")
    ap.add_argument("checkpoint", nargs="?", default="data/checkpoint_step_1000.pt",
                    help="checkpoint file (model weights) path")
    ap.add_argument("tokenizer", nargs="?", default="tokenizer.json",
                    help="token file (to init tokenizer) path")
    ap.add_argument("--system", type=str,
                    default="You are a helpful assistant.",
                    help="System prompt for chat mode")

    args = ap.parse_args()

    checkpoint_path = args.checkpoint
    tokenizer_path = args.tokenizer

    if not Path(checkpoint_path).exists():
        print(f"[Error] checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    if not Path(tokenizer_path).exists():
        print(f"[Error] tokenizer file not found: {tokenizer_path}")
        sys.exit(1)

    tok = tokenizer(Path(tokenizer_path))
    vocab_size = tok.vocab_size()

    d_model = 704 // 2
    head = 11
    n_layers = 30

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = gpt(vocab_size, d_model, head, n_layers).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])

    print(f"[Info] loaded {checkpoint_path} (step {ckpt['step']}) on {device}")
    print(f"[Info] type 'quit'/'exit' to exit, type 'clear' to start over")
    print(f"[Info] model: d={d_model} head={head} layers={n_layers}")
    print()

    chat_loop(model, tok, device, args.system)


if __name__ == "__main__":
    main()
