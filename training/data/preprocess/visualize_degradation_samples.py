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
        apply_degradation,
        derive_degradation_seed,
        parse_degradation_setting,
        sample_degradation_params,
    )
except ImportError:
    from data.degradation import (
        apply_degradation,
        derive_degradation_seed,
        parse_degradation_setting,
        sample_degradation_params,
    )


DEFAULT_VIS_SETTINGS = (
    "clean",
    "motion_blur_medium",
    "exposure_medium",
    "mixed_medium",
)


def visualize_degradation_samples(
    *,
    annotation_path: Path,
    data_root: Path,
    output_dir: Path,
    settings: Sequence[str] = DEFAULT_VIS_SETTINGS,
    num_samples: int = 8,
    base_seed: int = 42,
    overwrite: bool = True,
) -> Dict:
    annotation = _load_annotation(Path(annotation_path))
    frame_entries = list(_iter_limited_frames(annotation, max_frames=num_samples))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict] = []
    summary = {
        "annotation": str(annotation_path),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "settings": list(settings),
        "requested_samples": int(num_samples),
        "written_samples": 0,
        "written_images": 0,
        "missing_clean_images": 0,
    }

    for sample_index, (sequence_key, sequence, frame) in enumerate(frame_entries):
        clean_rel_path = Path(frame.get("clean_image_rel_path") or frame["image_rel_path"])
        clean_path = Path(data_root) / clean_rel_path
        image = cv2.imread(str(clean_path), cv2.IMREAD_COLOR)
        if image is None:
            summary["missing_clean_images"] += 1
            continue

        sample_dir = output_dir if len(frame_entries) == 1 else output_dir / f"sample_{sample_index:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        sample_records = []
        for setting in settings:
            output_path = sample_dir / f"{setting}.png"
            if output_path.is_file() and not overwrite:
                continue

            if setting == "clean":
                output_image = image
                metadata = {
                    "degradation_type": "clean",
                    "severity": "none",
                    "seed": None,
                    "params": {},
                }
            else:
                degradation_type, severity = parse_degradation_setting(setting)
                seed = derive_degradation_seed(
                    base_seed=base_seed,
                    seq_name=str(sequence.get("sequence_name") or sequence_key),
                    frame_id=int(frame.get("frame_id", 0)),
                    epoch=0,
                    degradation_type=degradation_type,
                )
                config = sample_degradation_params(degradation_type, severity, seed)
                output_image, metadata = apply_degradation(image, config)

            if not cv2.imwrite(str(output_path), output_image):
                raise IOError(f"Failed to write visualization image: {output_path}")

            record = {
                "sample_index": sample_index,
                "dataset": sequence.get("dataset", "euroc"),
                "sequence_name": sequence.get("sequence_name"),
                "camera_name": sequence.get("camera_name"),
                "frame_id": int(frame.get("frame_id", 0)),
                "timestamp_ns": int(frame.get("timestamp_ns", 0)),
                "clean_image_rel_path": clean_rel_path.as_posix(),
                "output_image": output_path.name,
                "setting": setting,
                "degradation_type": metadata["degradation_type"],
                "severity": metadata["severity"],
                "seed": metadata["seed"],
                "params": metadata["params"],
            }
            sample_records.append(record)
            all_records.append(record)
            summary["written_images"] += 1

        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as fout:
            json.dump(sample_records, fout, indent=2, sort_keys=True)
        summary["written_samples"] += 1

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fout:
        json.dump(all_records, fout, indent=2, sort_keys=True)

    return summary


def _load_annotation(annotation_path: Path) -> Dict:
    with gzip.open(annotation_path, "rt", encoding="utf-8") as fin:
        return json.load(fin)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write clean/degraded image samples for visual sanity checks."
    )
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument(
        "--settings",
        nargs="+",
        default=list(DEFAULT_VIS_SETTINGS),
    )
    parser.add_argument("--no_overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = visualize_degradation_samples(
        annotation_path=args.annotation,
        data_root=args.data_root,
        output_dir=args.output_dir,
        settings=args.settings,
        num_samples=args.num_samples,
        base_seed=args.base_seed,
        overwrite=not args.no_overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
