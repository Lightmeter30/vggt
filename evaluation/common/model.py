import random

import numpy as np
import torch

from vggt.models.vggt import VGGT


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg=None):
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_inference_dtype(device):
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.get_device_capability(device=device)[0] >= 8:
        return torch.bfloat16
    return torch.float16


def _looks_like_state_dict(value):
    return isinstance(value, dict) and bool(value) and all(
        torch.is_tensor(tensor) for tensor in value.values()
    )


def _strip_state_dict_prefixes(state_dict):
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key
        for prefix in ("module.", "_orig_mod."):
            if cleaned_key.startswith(prefix):
                cleaned_key = cleaned_key[len(prefix) :]
        cleaned_state_dict[cleaned_key] = value
    return cleaned_state_dict


def _extract_state_dict(checkpoint):
    if _looks_like_state_dict(checkpoint):
        return checkpoint

    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            candidate = checkpoint.get(key)
            if _looks_like_state_dict(candidate):
                return candidate

    raise ValueError(
        "Unsupported checkpoint format. Expected a state dict or a dict containing "
        "'model', 'state_dict', or 'model_state_dict'."
    )


def load_model(device, model_path):
    print("Initializing and loading VGGT model...")
    model = VGGT()
    print(f"USING {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = _strip_state_dict_prefixes(_extract_state_dict(checkpoint))
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)
