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


def _infer_imu_config_from_state_dict(state_dict):
    if not any(key.startswith("imu_encoder.") for key in state_dict):
        return None

    input_weight = state_dict.get("imu_encoder.input_proj.weight")
    output_weight = state_dict.get("imu_encoder.output_proj.weight")
    if input_weight is None or output_weight is None:
        raise ValueError("IMU checkpoint is missing imu_encoder input/output projection weights.")

    layer_indices = []
    prefix = "imu_encoder.temporal_encoder.layers."
    for key in state_dict:
        if key.startswith(prefix):
            rest = key[len(prefix):]
            layer_index = rest.split(".", 1)[0]
            if layer_index.isdigit():
                layer_indices.append(int(layer_index))

    return {
        "enabled": True,
        "input_dim": int(input_weight.shape[1]),
        "hidden_dim": int(input_weight.shape[0]),
        "num_layers": max(layer_indices) + 1 if layer_indices else 2,
        "num_heads": 4,
        "dropout": 0.1,
    }


def _infer_fusion_config_from_state_dict(state_dict):
    if not any(key.startswith("imu_fusion.") for key in state_dict):
        return None
    return {"enabled": True, "type": "film"}


def build_model_from_state_dict(state_dict):
    imu_config = _infer_imu_config_from_state_dict(state_dict)
    fusion_config = _infer_fusion_config_from_state_dict(state_dict)
    return VGGT(
        enable_camera=any(key.startswith("camera_head.") for key in state_dict),
        enable_point=any(key.startswith("point_head.") for key in state_dict),
        enable_depth=any(key.startswith("depth_head.") for key in state_dict),
        enable_track=any(key.startswith("track_head.") for key in state_dict),
        imu=imu_config,
        fusion=fusion_config,
    )


def load_model(device, model_path):
    print("Initializing and loading VGGT model...")
    print(f"USING {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = _strip_state_dict_prefixes(_extract_state_dict(checkpoint))
    model = build_model_from_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)
