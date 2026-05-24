import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from training.data.degradation import (
        SCHEMA_VERSION,
        apply_degradation,
        derive_degradation_seed,
        parse_degradation_setting,
        sample_degradation_params,
    )
except ImportError:
    from data.degradation import (
        SCHEMA_VERSION,
        apply_degradation,
        derive_degradation_seed,
        parse_degradation_setting,
        sample_degradation_params,
    )


DEFAULT_FIXED_SETTINGS = (
    "motion_blur_medium",
    "exposure_medium",
    "mixed_medium",
)


def load_annotation(annotation_path: Path) -> Dict:
    with gzip.open(annotation_path, "rt", encoding="utf-8") as fin:
        return json.load(fin)


def generate_degraded_images(
    *,
    annotation_path: Path,
    data_root: Path,
    output_root: Path,
    settings: Sequence[str] = DEFAULT_FIXED_SETTINGS,
    max_frames: Optional[int] = None,
    dry_run: bool = False,
    overwrite: bool = False,
    base_seed: int = 42,
) -> Dict:
    """按 annotation 固定生成退化图像和 JSONL metadata。"""
    annotation_path = Path(annotation_path)
    data_root = Path(data_root)
    output_root = Path(output_root)
    settings = [str(setting) for setting in settings]

    annotation = load_annotation(annotation_path)
    frame_entries = list(_iter_limited_frames(annotation, max_frames=max_frames))

    summary = {
        "annotation": str(annotation_path),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "settings": settings,
        "candidate_frames": len(frame_entries),
        "candidate_images": len(frame_entries) * len(settings),
        "written_images": 0,
        "skipped_existing": 0,
        "missing_clean_images": 0,
        "metadata_records": 0,
        "dry_run": bool(dry_run),
    }

    if dry_run:
        for _, sequence, frame in frame_entries:
            clean_rel_path = _frame_clean_rel_path(frame)
            if not (data_root / clean_rel_path).is_file():
                summary["missing_clean_images"] += len(settings)
        return summary

    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "degradation_metadata.jsonl"

    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        for sequence_key, sequence, frame in frame_entries:
            for setting in settings:
                clean_rel_path = _frame_clean_rel_path(frame)
                clean_path = data_root / clean_rel_path
                if not clean_path.is_file():
                    summary["missing_clean_images"] += 1
                    continue

                degradation_type, severity = parse_degradation_setting(setting)
                seed = derive_degradation_seed(
                    base_seed=base_seed,
                    seq_name=str(sequence.get("sequence_name") or sequence_key),
                    frame_id=int(frame.get("frame_id", 0)),
                    epoch=0,
                    degradation_type=degradation_type,
                )
                degradation_config = sample_degradation_params(
                    degradation_type=degradation_type,
                    severity=severity,
                    seed=seed,
                )

                degraded_rel_path = _build_degraded_rel_path(
                    setting=setting,
                    sequence=sequence,
                    sequence_key=sequence_key,
                    frame=frame,
                    clean_rel_path=clean_rel_path,
                )
                degraded_path = output_root / degraded_rel_path
                should_write = overwrite or not degraded_path.is_file()

                if should_write:
                    image = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
                    if image is None:
                        summary["missing_clean_images"] += 1
                        continue
                    degraded, degradation_metadata = apply_degradation(
                        image,
                        degradation_config,
                    )
                    degraded_path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(degraded_path), degraded):
                        raise IOError(f"Failed to write degraded image: {degraded_path}")
                    summary["written_images"] += 1
                else:
                    degradation_metadata = {
                        "schema_version": SCHEMA_VERSION,
                        "degradation_type": degradation_config["type"],
                        "severity": degradation_config["severity"],
                        "seed": degradation_config["seed"],
                        "params": degradation_config["params"],
                    }
                    summary["skipped_existing"] += 1

                record = _build_metadata_record(
                    sequence=sequence,
                    frame=frame,
                    clean_rel_path=clean_rel_path,
                    degraded_rel_path=degraded_rel_path,
                    setting=setting,
                    degradation_metadata=degradation_metadata,
                )
                metadata_file.write(json.dumps(record, sort_keys=True) + "\n")
                summary["metadata_records"] += 1

    return summary


def _iter_limited_frames(
    annotation: Mapping[str, Mapping],
    max_frames: Optional[int],
) -> Iterable[Tuple[str, Mapping, Mapping]]:
    yielded = 0
    for sequence_key, sequence in sorted(annotation.items()):
        for frame in sequence.get("frames", []):
            if max_frames is not None and yielded >= int(max_frames):
                return
            yielded += 1
            yield sequence_key, sequence, frame


def _frame_clean_rel_path(frame: Mapping) -> Path:
    return Path(frame.get("clean_image_rel_path") or frame["image_rel_path"])


def _build_degraded_rel_path(
    *,
    setting: str,
    sequence: Mapping,
    sequence_key: str,
    frame: Mapping,
    clean_rel_path: Path,
) -> Path:
    sequence_name = str(sequence.get("sequence_name") or sequence_key).strip("/")
    camera_name = str(sequence.get("camera_name") or "camera").strip("/")
    timestamp_ns = int(frame.get("timestamp_ns", frame.get("frame_id", 0)))
    suffix = clean_rel_path.suffix or ".png"
    return Path(setting) / sequence_name / camera_name / f"{timestamp_ns}{suffix}"


def _build_metadata_record(
    *,
    sequence: Mapping,
    frame: Mapping,
    clean_rel_path: Path,
    degraded_rel_path: Path,
    setting: str,
    degradation_metadata: Mapping,
) -> Dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": sequence.get("dataset", "euroc"),
        "sequence_name": sequence.get("sequence_name"),
        "camera_name": sequence.get("camera_name"),
        "split": sequence.get("split"),
        "frame_id": int(frame.get("frame_id", 0)),
        "timestamp_ns": int(frame.get("timestamp_ns", 0)),
        "clean_image_rel_path": clean_rel_path.as_posix(),
        "degraded_image_rel_path": degraded_rel_path.as_posix(),
        "setting": setting,
        "degradation_type": degradation_metadata["degradation_type"],
        "severity": degradation_metadata["severity"],
        "seed": degradation_metadata["seed"],
        "params": degradation_metadata["params"],
        "source_pose_unchanged": True,
        "source_intrinsics_unchanged": True,
        "source_imu_unchanged": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed degraded images and JSONL metadata from VI annotations."
    )
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument(
        "--settings",
        nargs="+",
        default=list(DEFAULT_FIXED_SETTINGS),
        help="Degradation settings such as motion_blur_medium exposure_medium mixed_medium.",
    )
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_degraded_images(
        annotation_path=args.annotation,
        data_root=args.data_root,
        output_root=args.output_root,
        settings=args.settings,
        max_frames=args.max_frames,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        base_seed=args.base_seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
