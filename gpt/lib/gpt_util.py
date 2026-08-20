def format_token(token_size: int) -> str:
    if token_size < 1e3:
        return f"{token_size}"
    elif token_size < 1e6:
        return f"{token_size / 1e3:.2f}k"
    elif token_size < 1e9:
        return f"{token_size / 1e6:.2f}M"
    return f"{token_size / 1e9:.2f}B"