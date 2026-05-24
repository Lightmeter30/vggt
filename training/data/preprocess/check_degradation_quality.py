import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Mapping, Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from training.data.degradation import parse_degradation_setting
except ImportError:
    from data.degradation import parse_degradation_setting

try:
    from skimage.metrics import structural_similarity as structural_similarity
except ImportError:
    structural_similarity = None


PSNR_WARNING_RANGES = {
    "motion_blur_medium": (18.0, 40.0),
    "exposure_medium": (12.0, 35.0),
    "mixed_medium": (10.0, 35.0),
}


def check_degradation_quality(
    *,
    metadata_path: Path,
    data_root: Path,
    degraded_root: Path,
    max_samples_per_setting: Optional[int] = 100,
) -> Dict:
    records = _read_jsonl(Path(metadata_path))
    grouped_values: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"psnr": [], "ssim": []}
    )
    warnings: List[str] = []
    seen_per_setting: Dict[str, int] = defaultdict(int)

    for record in records:
        setting = str(record.get("setting") or _setting_from_record(record))
        if max_samples_per_setting is not None and seen_per_setting[setting] >= int(max_samples_per_setting):
            continue
        seen_per_setting[setting] += 1

        clean_path = _resolve_path(Path(data_root), record["clean_image_rel_path"])
        degraded_path = _resolve_path(Path(degraded_root), record["degraded_image_rel_path"])
        clean = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
        degraded = cv2.imread(str(degraded_path), cv2.IMREAD_COLOR)
        if clean is None:
            warnings.append(f"{setting}: missing clean image {clean_path}")
            continue
        if degraded is None:
            warnings.append(f"{setting}: missing degraded image {degraded_path}")
            continue
        if clean.shape != degraded.shape:
            warnings.append(
                f"{setting}: shape mismatch clean={clean.shape} degraded={degraded.shape}"
            )
            continue

        psnr = compute_psnr(clean, degraded)
        grouped_values[setting]["psnr"].append(psnr)
        if structural_similarity is not None:
            grouped_values[setting]["ssim"].append(compute_ssim(clean, degraded))

        if np.array_equal(clean, degraded) and record.get("degradation_type") != "clean":
            warnings.append(f"{setting}: degraded image is identical to clean image")
        if degraded.max() <= 2:
            warnings.append(f"{setting}: degraded image is almost all black")
        if degraded.min() >= 253:
            warnings.append(f"{setting}: degraded image is almost all white")

    setting_stats = {
        setting: _summarize_metrics(values)
        for setting, values in sorted(grouped_values.items())
    }
    warnings.extend(_build_stat_warnings(setting_stats))

    summary = {
        "metadata": str(metadata_path),
        "data_root": str(data_root),
        "degraded_root": str(degraded_root),
        "num_records": len(records),
        "num_settings": len(setting_stats),
        "ssim_available": structural_similarity is not None,
        "settings": setting_stats,
        "warnings": warnings,
    }
    return summary


def compute_psnr(clean: np.ndarray, degraded: np.ndarray) -> float:
    diff = clean.astype(np.float32) - degraded.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return math.inf
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def compute_ssim(clean: np.ndarray, degraded: np.ndarray) -> float:
    if structural_similarity is None:
        return math.nan
    try:
        return float(
            structural_similarity(
                clean,
                degraded,
                channel_axis=-1,
                data_range=255,
            )
        )
    except TypeError:
        return float(
            structural_similarity(
                clean,
                degraded,
                multichannel=True,
                data_range=255,
            )
        )


def _read_jsonl(path: Path) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                records.append(json.loads(line))
    return records


def _resolve_path(root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return root / path


def _setting_from_record(record: Mapping) -> str:
    degradation_type = record.get("degradation_type", "clean")
    severity = record.get("severity", "none")
    if degradation_type == "clean" or severity == "none":
        return "clean"
    return f"{degradation_type}_{severity}"


def _summarize_metrics(values: Mapping[str, List[float]]) -> Dict:
    stats = {
        "count": len(values["psnr"]),
        "psnr": _stats(values["psnr"]),
    }
    if values["ssim"]:
        stats["ssim"] = _stats(values["ssim"])
    return stats


def _stats(values: List[float]) -> Dict:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return {"mean": math.inf, "median": math.inf, "min": math.inf, "max": math.inf}
    return {
        "mean": float(mean(finite_values)),
        "median": float(median(finite_values)),
        "min": float(min(finite_values)),
        "max": float(max(finite_values)),
    }


def _build_stat_warnings(setting_stats: Mapping[str, Mapping]) -> List[str]:
    warnings = []
    for setting, (low, high) in PSNR_WARNING_RANGES.items():
        psnr_mean = _metric_mean(setting_stats, setting, "psnr")
        if psnr_mean is None:
            continue
        if psnr_mean < low or psnr_mean > high:
            warnings.append(
                f"{setting}: mean PSNR {psnr_mean:.2f} dB is outside diagnostic range [{low}, {high}]"
            )

    for degradation_type in ("motion_blur", "exposure", "mixed"):
        mild = _metric_mean(setting_stats, f"{degradation_type}_mild", "psnr")
        medium = _metric_mean(setting_stats, f"{degradation_type}_medium", "psnr")
        strong = _metric_mean(setting_stats, f"{degradation_type}_strong", "psnr")
        if mild is not None and medium is not None and mild <= medium:
            warnings.append(
                f"{degradation_type}: mild PSNR should be higher than medium"
            )
        if medium is not None and strong is not None and medium <= strong:
            warnings.append(
                f"{degradation_type}: medium PSNR should be higher than strong"
            )

    mixed_medium = _metric_mean(setting_stats, "mixed_medium", "psnr")
    motion_medium = _metric_mean(setting_stats, "motion_blur_medium", "psnr")
    exposure_medium = _metric_mean(setting_stats, "exposure_medium", "psnr")
    if (
        mixed_medium is not None
        and motion_medium is not None
        and exposure_medium is not None
        and mixed_medium > max(motion_medium, exposure_medium)
    ):
        warnings.append("mixed_medium: PSNR is higher than both single degradations")

    return warnings


def _metric_mean(
    setting_stats: Mapping[str, Mapping],
    setting: str,
    metric_name: str,
) -> Optional[float]:
    if setting not in setting_stats:
        return None
    value = setting_stats[setting].get(metric_name, {}).get("mean")
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PSNR and optional SSIM diagnostics for degraded images."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--degraded_root", type=Path, required=True)
    parser.add_argument("--max_samples_per_setting", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = check_degradation_quality(
        metadata_path=args.metadata,
        data_root=args.data_root,
        degraded_root=args.degraded_root,
        max_samples_per_setting=args.max_samples_per_setting,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
