import torch

def get_device() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        print("[Info] CUDA device available")
        return torch.device("cuda"), "cuda"
    elif torch.backends.mps.is_available():
        print("[Info] MPS device available")
        return torch.device("mps"), "mps"
    else:
        print("[Warning] No CUDA or MPS device available, using CPU instead")
        return torch.device("cpu"), "cpu"

def empty_cache(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    # cpu does not have a cache

def torch_amp_available(device_name: str) -> bool:
    return device_name in ("cuda", "mps")