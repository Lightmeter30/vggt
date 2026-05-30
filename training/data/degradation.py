import hashlib
from copy import deepcopy
from typing import Dict, Mapping, Optional, Tuple

import cv2
import numpy as np


SCHEMA_VERSION = "degradation_v1"
DEGRADATION_LABEL_TO_ID = {
    "clean": 0,
    "motion_blur": 1,
    "exposure": 2,
    "mixed": 3,
}

DEFAULT_TRAINING_DEGRADATION_WEIGHTS = {
    "clean": 0.25,
    "motion_blur": 0.25,
    "exposure": 0.25,
    "mixed": 0.25,
}

SEVERITY_SPECS = {
    "mild": {
        "motion_kernel": 7,
        "gain_range": (0.7, 1.3),
        "gamma_range": (0.85, 1.15),
    },
    "medium": {
        "motion_kernel": 15,
        "gain_range": (0.45, 1.8),
        "gamma_range": (0.65, 1.45),
    },
    "strong": {
        "motion_kernel": 25,
        "gain_range": (0.25, 2.5),
        "gamma_range": (0.45, 1.8),
    },
}


def derive_degradation_seed(
    base_seed: int,
    seq_name: str,
    frame_id: int,
    epoch: int,
    degradation_type: str,
) -> int:
    """派生确定性 31-bit seed，避免 Python 内置 hash 的随机化。"""
    # Python 内置 hash() 受 PYTHONHASHSEED 影响，不适合作为可复现实验 seed。
    # seq_name/frame_id/epoch/degradation_type 都进入 key，避免跨序列、跨轮次或
    # 不同退化类型在同一帧上碰撞；离线 val/test 固定使用 epoch=0。
    key = f"{int(base_seed)}|{seq_name}|{int(frame_id)}|{int(epoch)}|{degradation_type}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31)


def build_motion_blur_kernel(kernel_size: int, angle_deg: float) -> np.ndarray:
    """构造归一化线性运动模糊核。"""
    kernel_size = int(kernel_size)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

    center = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[center, :] = 1.0

    matrix = cv2.getRotationMatrix2D(
        (float(center), float(center)),
        float(angle_deg) % 180.0,
        1.0,
    )
    rotated = cv2.warpAffine(
        kernel,
        matrix,
        (kernel_size, kernel_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ).astype(np.float32)

    kernel_sum = float(rotated.sum())
    if kernel_sum <= 0.0:
        rotated = kernel
        kernel_sum = float(rotated.sum())

    return (rotated / kernel_sum).astype(np.float32)


def apply_motion_blur(
    image: np.ndarray,
    kernel_size: int,
    angle_deg: float,
) -> np.ndarray:
    """对图像应用线性运动模糊，不改变 shape 和 dtype。"""
    kernel = build_motion_blur_kernel(kernel_size, angle_deg)
    blurred = cv2.filter2D(
        image,
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT101,
    )
    return blurred.astype(image.dtype, copy=False)


def apply_exposure_variation(
    image: np.ndarray,
    gain: float,
    gamma: float,
    black_level: float = 0.0,
) -> np.ndarray:
    """应用全局曝光变化，输出裁剪到输入 dtype 对应范围。"""
    gain = float(gain)
    gamma = float(gamma)
    black_level = float(black_level)
    if gain <= 0.0:
        raise ValueError(f"gain must be positive, got {gain}")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}")

    max_value = _infer_image_max_value(image)
    work = image.astype(np.float32)
    normalized = np.clip((work - black_level) / max_value, 0.0, 1.0)
    adjusted = np.power(normalized, gamma) * gain * max_value + black_level
    adjusted = np.clip(adjusted, 0.0, max_value)
    return _restore_dtype(adjusted, image.dtype)


def sample_degradation_type(settings: Mapping[str, float], seed: int) -> str:
    """按权重确定性采样退化类型。"""
    if not settings:
        raise ValueError("settings must contain at least one degradation type")

    ordered_names = [
        name
        for name in DEGRADATION_LABEL_TO_ID
        if name in settings
    ]
    ordered_names.extend(sorted(name for name in settings if name not in ordered_names))

    names = []
    weights = []
    for name in ordered_names:
        weight = float(settings[name])
        if weight < 0.0:
            raise ValueError(f"degradation weight must be non-negative, got {name}={weight}")
        if weight > 0.0:
            names.append(name)
            weights.append(weight)

    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        raise ValueError("at least one degradation weight must be positive")

    probabilities = np.asarray(weights, dtype=np.float64) / total_weight
    rng = np.random.default_rng(int(seed))
    selected = int(rng.choice(len(names), p=probabilities))
    degradation_type, severity = parse_degradation_setting(names[selected], default_severity=None)
    return _validate_degradation_type(degradation_type, allow_clean=True)


def sample_degradation_params(
    degradation_type: str,
    severity: str,
    seed: Optional[int],
) -> Dict:
    """根据类型、强度和 seed 采样可复现退化参数。"""
    degradation_type, parsed_severity = parse_degradation_setting(
        degradation_type,
        default_severity=severity,
    )
    if degradation_type == "clean":
        return {
            "type": "clean",
            "severity": "none",
            "seed": None,
            "params": {},
        }

    severity = parsed_severity or severity
    spec = _get_severity_spec(severity)
    rng = np.random.default_rng(None if seed is None else int(seed))

    if degradation_type == "motion_blur":
        params = _sample_motion_params(rng, spec)
    elif degradation_type == "exposure":
        params = _sample_exposure_params(rng, spec)
    elif degradation_type == "mixed":
        params = {
            "order": ["motion_blur", "exposure"],
            "motion_blur": _sample_motion_params(rng, spec),
            "exposure": _sample_exposure_params(rng, spec),
        }
    else:
        raise ValueError(f"Unsupported degradation_type: {degradation_type}")

    return {
        "type": degradation_type,
        "severity": severity,
        "seed": None if seed is None else int(seed),
        "params": params,
    }


def apply_degradation(image: np.ndarray, config: Mapping) -> Tuple[np.ndarray, Dict]:
    """应用退化配置，并返回退化图像和 metadata。"""
    config = _normalize_config(config)
    degradation_type, severity = parse_degradation_setting(
        config.get("degradation_type", config.get("type", "clean")),
        default_severity=config.get("severity"),
    )
    seed = config.get("seed")

    if "params" in config and config["params"] is not None:
        params = deepcopy(config["params"])
    else:
        sampled = sample_degradation_params(degradation_type, severity or "medium", seed)
        severity = sampled["severity"]
        params = deepcopy(sampled["params"])

    if degradation_type == "clean":
        severity = "none"
        output = np.array(image, copy=True)
        params = {}
    elif degradation_type == "motion_blur":
        output = apply_motion_blur(
            image,
            kernel_size=int(params["kernel_size"]),
            angle_deg=float(params["angle_deg"]),
        )
    elif degradation_type == "exposure":
        output = apply_exposure_variation(
            image,
            gain=float(params["gain"]),
            gamma=float(params["gamma"]),
            black_level=float(params.get("black_level", 0.0)),
        )
    elif degradation_type == "mixed":
        order = params.get("order", ["motion_blur", "exposure"])
        # 各子步骤（motion_blur / exposure）均返回新数组，无需提前拷贝
        output = image
        for step in order:
            if step == "motion_blur":
                motion_params = params["motion_blur"]
                output = apply_motion_blur(
                    output,
                    kernel_size=int(motion_params["kernel_size"]),
                    angle_deg=float(motion_params["angle_deg"]),
                )
            elif step == "exposure":
                exposure_params = params["exposure"]
                output = apply_exposure_variation(
                    output,
                    gain=float(exposure_params["gain"]),
                    gamma=float(exposure_params["gamma"]),
                    black_level=float(exposure_params.get("black_level", 0.0)),
                )
            else:
                raise ValueError(f"Unsupported mixed degradation step: {step}")
        params["order"] = list(order)
    else:
        raise ValueError(f"Unsupported degradation_type: {degradation_type}")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "degradation_type": degradation_type,
        "severity": severity,
        "seed": None if seed is None else int(seed),
        "params": params,
    }
    return output.astype(image.dtype, copy=False), metadata


def parse_degradation_setting(
    setting: str,
    default_severity: Optional[str] = "medium",
) -> Tuple[str, str]:
    """解析 clean、motion_blur_medium 等设置名。"""
    if setting is None:
        return "clean", "none"

    setting = str(setting)
    if setting == "clean":
        return "clean", "none"

    for degradation_type in ("motion_blur", "exposure", "mixed"):
        prefix = f"{degradation_type}_"
        if setting == degradation_type:
            severity = default_severity or "medium"
            _get_severity_spec(severity)
            return degradation_type, severity
        if setting.startswith(prefix):
            severity = setting[len(prefix):]
            _get_severity_spec(severity)
            return degradation_type, severity

    raise ValueError(f"Unsupported degradation setting: {setting}")


def _sample_motion_params(rng: np.random.Generator, spec: Mapping) -> Dict:
    return {
        "kernel_size": int(spec["motion_kernel"]),
        "angle_deg": float(rng.uniform(0.0, 180.0)),
        "direction_source": "random",
        "border_mode": "reflect101",
    }


def _sample_exposure_params(rng: np.random.Generator, spec: Mapping) -> Dict:
    gain_min, gain_max = spec["gain_range"]
    gamma_min, gamma_max = spec["gamma_range"]
    return {
        "gain": float(rng.uniform(gain_min, gain_max)),
        "gamma": float(rng.uniform(gamma_min, gamma_max)),
        "black_level": 0.0,
        "clip": [0, 255],
    }


def _infer_image_max_value(image: np.ndarray) -> float:
    if np.issubdtype(image.dtype, np.integer):
        return float(np.iinfo(image.dtype).max)

    if image.size == 0:
        return 1.0
    finite_max = float(np.nanmax(image))
    return 1.0 if finite_max <= 1.0 else 255.0


def _restore_dtype(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        max_value = np.iinfo(dtype).max
        min_value = np.iinfo(dtype).min
        return np.clip(np.rint(image), min_value, max_value).astype(dtype)
    return image.astype(dtype, copy=False)


def _get_severity_spec(severity: str) -> Dict:
    if severity not in SEVERITY_SPECS:
        raise ValueError(
            f"Unsupported severity: {severity}. Expected one of {sorted(SEVERITY_SPECS)}"
        )
    return SEVERITY_SPECS[severity]


def _validate_degradation_type(degradation_type: str, allow_clean: bool) -> str:
    valid_types = set(DEGRADATION_LABEL_TO_ID)
    if not allow_clean:
        valid_types.remove("clean")
    if degradation_type not in valid_types:
        raise ValueError(
            f"Unsupported degradation_type: {degradation_type}. "
            f"Expected one of {sorted(valid_types)}"
        )
    return degradation_type


def _normalize_config(config: Mapping) -> Dict:
    if config is None:
        return {"type": "clean", "severity": "none", "seed": None, "params": {}}
    if hasattr(config, "items"):
        return dict(config.items())
    return dict(config)
