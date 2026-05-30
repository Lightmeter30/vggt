import random
from collections.abc import Mapping

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


def _extract_model_config(checkpoint):
    if not isinstance(checkpoint, dict) or _looks_like_state_dict(checkpoint):
        return None
    if "model_config" in checkpoint:
        return checkpoint["model_config"]
    config = checkpoint.get("config")
    if isinstance(config, Mapping) and "model" in config:
        return config["model"]
    return None


def _config_get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _has_config_key(config, key):
    if config is None:
        return False
    if isinstance(config, Mapping):
        return key in config
    return hasattr(config, key)


def _validate_config_matches_state(name, configured_value, inferred_value):
    if configured_value is None:
        return
    if int(configured_value) != int(inferred_value):
        raise ValueError(
            f"Checkpoint model_config {name}={configured_value} does not match "
            f"state_dict-inferred {name}={inferred_value}."
        )


def _infer_imu_config_from_state_dict(state_dict, model_config=None):
    if not any(key.startswith("imu_encoder.") for key in state_dict):
        return None
    if model_config is None:
        raise ValueError(
            "IMU checkpoint requires checkpoint['model_config'] with imu.num_heads "
            "and imu.dropout. These values cannot be inferred safely from state_dict."
        )

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

    inferred_config = {
        "input_dim": int(input_weight.shape[1]),
        "hidden_dim": int(input_weight.shape[0]),
        "num_layers": max(layer_indices) + 1 if layer_indices else 2,
    }
    imu_config = _config_get(model_config, "imu", None)
    _validate_config_matches_state("imu.input_dim", _config_get(imu_config, "input_dim", None), inferred_config["input_dim"])
    _validate_config_matches_state("imu.hidden_dim", _config_get(imu_config, "hidden_dim", None), inferred_config["hidden_dim"])
    _validate_config_matches_state("imu.num_layers", _config_get(imu_config, "num_layers", None), inferred_config["num_layers"])

    if imu_config is None or not _has_config_key(imu_config, "num_heads") or not _has_config_key(imu_config, "dropout"):
        raise ValueError(
            "IMU checkpoint model_config must include imu.num_heads and imu.dropout."
        )

    return {
        "enabled": True,
        "input_dim": int(_config_get(imu_config, "input_dim", inferred_config["input_dim"])),
        "hidden_dim": int(_config_get(imu_config, "hidden_dim", inferred_config["hidden_dim"])),
        "num_layers": int(_config_get(imu_config, "num_layers", inferred_config["num_layers"])),
        "num_heads": int(_config_get(imu_config, "num_heads", 4)),
        "dropout": float(_config_get(imu_config, "dropout", 0.1)),
    }


def _infer_fusion_hidden_dim_from_state_dict(state_dict):
    candidates = []
    first_linear = state_dict.get("imu_fusion.film.1.weight")
    second_linear = state_dict.get("imu_fusion.film.3.weight")
    if first_linear is not None:
        candidates.append(int(first_linear.shape[0]))
    if second_linear is not None:
        candidates.append(int(second_linear.shape[1]))
    if not candidates:
        return None
    if any(candidate != candidates[0] for candidate in candidates):
        raise ValueError("IMU fusion checkpoint has inconsistent hidden_dim weights.")
    return candidates[0]


def _infer_fusion_config_from_state_dict(state_dict, model_config=None):
    if not any(key.startswith("imu_fusion.") for key in state_dict):
        return None

    inferred_hidden_dim = _infer_fusion_hidden_dim_from_state_dict(state_dict)
    fusion_config = _config_get(model_config, "fusion", None)
    if fusion_config is None:
        raise ValueError("IMU fusion checkpoint requires checkpoint['model_config']['fusion'].")
    configured_hidden_dim = _config_get(fusion_config, "hidden_dim", None)
    if configured_hidden_dim is not None and inferred_hidden_dim is not None:
        _validate_config_matches_state("fusion.hidden_dim", configured_hidden_dim, inferred_hidden_dim)

    return {
        "enabled": True,
        "type": str(_config_get(fusion_config, "type", "film")),
        "hidden_dim": configured_hidden_dim if configured_hidden_dim is not None else inferred_hidden_dim,
        "zero_init_gamma_scale": float(_config_get(fusion_config, "zero_init_gamma_scale", 1.0)),
        "zero_init_beta_scale": float(_config_get(fusion_config, "zero_init_beta_scale", 1.0)),
        "insert_at": str(_config_get(fusion_config, "insert_at", "aggregator_input")),
    }


def _resolve_head_enabled(model_config, key, state_dict, prefix):
    configured = _config_get(model_config, key, None)
    if configured is not None:
        return bool(configured)
    return any(state_key.startswith(prefix) for state_key in state_dict)


def _model_init_kwargs_from_config(model_config):
    kwargs = {}
    for key in ("img_size", "patch_size", "embed_dim"):
        value = _config_get(model_config, key, None)
        if value is not None:
            kwargs[key] = int(value)
    return kwargs


def build_model_from_state_dict(state_dict, model_config=None):
    imu_config = _infer_imu_config_from_state_dict(state_dict, model_config=model_config)
    fusion_config = _infer_fusion_config_from_state_dict(state_dict, model_config=model_config)
    model_kwargs = _model_init_kwargs_from_config(model_config)
    return VGGT(
        enable_camera=_resolve_head_enabled(model_config, "enable_camera", state_dict, "camera_head."),
        enable_point=_resolve_head_enabled(model_config, "enable_point", state_dict, "point_head."),
        enable_depth=_resolve_head_enabled(model_config, "enable_depth", state_dict, "depth_head."),
        enable_track=_resolve_head_enabled(model_config, "enable_track", state_dict, "track_head."),
        imu=imu_config,
        fusion=fusion_config,
        **model_kwargs,
    )


def load_model(device, model_path):
    print("Initializing and loading VGGT model...")
    print(f"USING {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    model_config = _extract_model_config(checkpoint)
    state_dict = _strip_state_dict_prefixes(_extract_state_dict(checkpoint))
    model = build_model_from_state_dict(state_dict, model_config=model_config)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)
