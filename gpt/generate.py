import torch
from pathlib import Path
from gpt import gpt
from tokenizer import tokenizer
import codecs
from lib.model_config import model_config

def generate_chat(model: gpt, tok: tokenizer,
                  ids: list[int], max_new_tokens: int = 512,
                  temperature: float = 0.8, top_k: int = 50,
                  repetition_penalty: float = 1.15, device=None):
    model.eval()

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    prompt_len = input_ids.shape[1]
    gen_count = 0
    dec = codecs.getincrementaldecoder("utf-8")(errors="ignore")

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # crop to max_seq_len if needed (take last N tokens)
            crop = input_ids[:, -1024:]

            logits = model(crop)      # (1, seq_len, vocab_size)
            logits = logits[:, -1, :] # (1, vocab_size) — last position only

            # repetition penalty: suppress tokens already generated
            if repetition_penalty > 1.0 and gen_count > 0:
                seen = input_ids[0, prompt_len:].tolist()
                if seen:
                    seen_ids = torch.tensor(seen, device=device)
                    logits[0, seen_ids] /= repetition_penalty

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
            gen_count += 1

            # stop generation if we generated <|im_end|>
            if next_id.item() == tok.vocab["<|im_end|>"]:
                break

            text = dec.decode(tok.decode_bytes([next_id.item()]))
            yield text

    dec.decode(b"", final=True)


def chat_loop(model: gpt, tok: tokenizer, device, system_prompt: str):
    messages = [{
        "role": "system",
        "content": system_prompt,
        "tokens": len(tok.encode(system_prompt))
    }]

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

        messages.append({
            "role": "user",
            "content": user_input,
            "tokens": len(tok.encode(user_input))
        })

        prompt = ""
        for m in messages:
            prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        ids = tok.encode(prompt)

        print("[Quetzal] ", end="", flush=True)
        response = ""
        for piece in generate_chat(model, tok, ids,
                                   max_new_tokens=512, temperature=0.8, top_k=50,
                                   repetition_penalty=1.15,
                                   device=device):
            if "\r" in piece:
                print(piece.replace("\r", "<|\\r|>"), end="", flush=True)
            else:
                print(piece, end="", flush=True)
            if piece.endswith("\n"):
                print("[Quetzal] ", end="", flush=True)
            response += piece
        print()

        messages.append({
            "role": "assistant",
            "content": response,
            "tokens": len(tok.encode(response))
        })

        curr_context_len = sum(len(m["content"]) for m in messages)

        do_clear = False
        while curr_context_len >= 1024:
            do_clear = True
            messages.pop(1)  # drop oldest non-system message
            curr_context_len = sum(len(m["content"]) for m in messages)
        if do_clear:
            print(f"[Quetzal] [Context usage: {curr_context_len / 1024:.2%} (cleared)]")
        else:
            print(f"[Quetzal] [Context usage: {curr_context_len / 1024:.2%}]")
        print()


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
    ap.add_argument("tokenizer", nargs="?", default="data/tokenizer.json",
                    help="token file (to init tokenizer) path, default: data/tokenizer.json")
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

    config = model_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = gpt(vocab_size, config.d_model, config.head, config.n_layers).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    step = ckpt['step']
    # release memory
    del ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Info] loaded {checkpoint_path} (step {step}) on {device}")
    print(f"[Info] type 'quit'/'exit' to exit, type 'clear' to start over")
    print(f"[Info] model: d={config.d_model} head={config.head} layers={config.n_layers}")
    print()

    chat_loop(model, tok, device, args.system)


if __name__ == "__main__":
    main()
