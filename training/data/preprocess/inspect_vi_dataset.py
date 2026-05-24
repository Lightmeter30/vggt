import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Sequence

try:
    from training.data.preprocess.generate_euroc_annotations import (
        discover_sequences,
        find_nearest_timestamp,
        load_camera_sensor,
        load_groundtruth,
        load_image_rows,
        load_imu_data,
    )
    from training.data.preprocess.vi_schema import (
        EUROC_SPLIT_SEQUENCES,
        build_split_manifest,
        normalize_split_sequences,
        sequence_to_split_roles,
        short_sequence_name,
    )
except ImportError:
    from generate_euroc_annotations import (
        discover_sequences,
        find_nearest_timestamp,
        load_camera_sensor,
        load_groundtruth,
        load_image_rows,
        load_imu_data,
    )
    from vi_schema import (
        EUROC_SPLIT_SEQUENCES,
        build_split_manifest,
        normalize_split_sequences,
        sequence_to_split_roles,
        short_sequence_name,
    )


def count_csv_rows(csv_path: Path) -> int:
    if not csv_path.is_file():
        return 0
    count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and not row[0].startswith("#"):
                count += 1
    return count


def inspect_euroc_sequence(
    euroc_dir: Path,
    sequence_dir: Path,
    camera_names: Sequence[str],
    max_pose_time_diff_ns: int,
) -> Dict:
    sequence_path = sequence_dir.relative_to(euroc_dir).as_posix()
    gt_csv = sequence_dir / "mav0" / "state_groundtruth_estimate0" / "data.csv"
    imu_csv = sequence_dir / "mav0" / "imu0" / "data.csv"

    gt_timestamps, _ = load_groundtruth(gt_csv)
    imu_count = count_csv_rows(imu_csv)

    per_camera = {}
    for camera_name in camera_names:
        camera_dir = sequence_dir / "mav0" / camera_name
        camera_csv = camera_dir / "data.csv"
        sensor_yaml = camera_dir / "sensor.yaml"
        image_rows = load_image_rows(camera_csv) if camera_csv.is_file() else []
        missing_images = 0
        matched_frames = 0
        pose_gap_skipped = 0
        pose_diffs = []

        for timestamp_ns, image_name in image_rows:
            image_path = camera_dir / "data" / image_name
            if not image_path.is_file():
                missing_images += 1
                continue
            _, pose_dt = find_nearest_timestamp(gt_timestamps, timestamp_ns)
            if pose_dt is None:
                continue
            pose_diffs.append(int(pose_dt))
            if pose_dt <= max_pose_time_diff_ns:
                matched_frames += 1
            else:
                pose_gap_skipped += 1

        sensor_ok = False
        sensor_fields = {}
        if sensor_yaml.is_file():
            intrinsics, distortion, body_from_camera = load_camera_sensor(sensor_yaml)
            sensor_ok = True
            sensor_fields = {
                "intrinsics_shape": list(intrinsics.shape),
                "distortion_len": int(len(distortion)),
                "T_imu_cam_shape": list(body_from_camera.shape),
            }

        per_camera[camera_name] = {
            "image_count": int(len(image_rows)),
            "matched_frames": int(matched_frames),
            "missing_images": int(missing_images),
            "pose_gap_skipped": int(pose_gap_skipped),
            "max_pose_time_diff_ns": int(max(pose_diffs)) if pose_diffs else None,
            "sensor_yaml": sensor_yaml.is_file(),
            "sensor_ok": sensor_ok,
            "sensor_fields": sensor_fields,
        }

    frame_count = 0
    if per_camera:
        frame_count = max(camera_stats["matched_frames"] for camera_stats in per_camera.values())

    return {
        "sequence_path": sequence_path,
        "sequence_name": short_sequence_name(sequence_path),
        "gt_count": int(len(gt_timestamps)),
        "imu_count": int(imu_count),
        "frame_count": int(frame_count),
        "per_camera": per_camera,
    }


def inspect_euroc(
    data_root: Path,
    camera_names: Sequence[str],
    max_pose_time_diff_ns: int,
) -> Dict:
    sequence_dirs = discover_sequences(data_root)
    split_sequences = normalize_split_sequences(EUROC_SPLIT_SEQUENCES)
    split_roles_by_sequence = sequence_to_split_roles(split_sequences)

    sequences = {}
    sequence_paths = {}
    frame_counts = {}
    for sequence_dir in sequence_dirs:
        stats = inspect_euroc_sequence(
            euroc_dir=data_root,
            sequence_dir=sequence_dir,
            camera_names=camera_names,
            max_pose_time_diff_ns=max_pose_time_diff_ns,
        )
        sequence_name = stats["sequence_name"]
        if sequence_name not in split_roles_by_sequence:
            raise ValueError(
                f"No EuRoC split configured for sequence {sequence_name} "
                f"({stats['sequence_path']})"
            )
        sequences[sequence_name] = stats
        sequence_paths[sequence_name] = stats["sequence_path"]
        frame_counts[sequence_name] = stats["frame_count"]

    manifest = build_split_manifest(
        dataset="euroc",
        sequence_paths=sequence_paths,
        frame_counts=frame_counts,
        camera_names=camera_names,
        max_pose_time_diff_ns=max_pose_time_diff_ns,
        split_sequences=split_sequences,
    )
    return {
        "dataset": "euroc",
        "data_root": data_root.as_posix(),
        "camera_names": list(camera_names),
        "sequence_count": len(sequences),
        "sequences": sequences,
        "split_manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect VI datasets without moving or deleting source data."
    )
    parser.add_argument("--dataset", choices=["euroc"], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--manifest_output",
        help="Path for split_manifest.json. Defaults to the output report directory.",
    )
    parser.add_argument("--camera_names", nargs="+", default=["cam0"])
    parser.add_argument("--max_pose_time_diff_ns", type=int, default=10_000_000)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dataset == "euroc":
        report = inspect_euroc(
            data_root=data_root,
            camera_names=args.camera_names,
            max_pose_time_diff_ns=args.max_pose_time_diff_ns,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    manifest_output = (
        Path(args.manifest_output).resolve()
        if args.manifest_output
        else output_path.parent / "split_manifest.json"
    )
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_output, "w", encoding="utf-8") as f:
        json.dump(report["split_manifest"], f, indent=2, ensure_ascii=False)

    print(f"Wrote inspection report: {output_path}")
    print(f"Wrote split manifest: {manifest_output}")
    print(f"Discovered sequences: {report['sequence_count']}")
    for split, counts in report["split_manifest"]["counts"].items():
        print(
            f"{split}: sequences={counts['sequence_count']} "
            f"frames={counts['frame_count']}"
        )


if __name__ == "__main__":
    main()
