import argparse
import csv
import gzip
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


def dump_jgz(path: Path, payload: Dict) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)


def discover_sequences(euroc_dir: Path) -> List[Path]:
    sequence_dirs: List[Path] = []

    for root, dirs, _ in os.walk(euroc_dir, followlinks=True):
        dirs[:] = [d for d in dirs if d != "__MACOSX"]
        root_path = Path(root)
        mav0_dir = root_path / "mav0"

        if not mav0_dir.is_dir():
            continue

        cam0_csv = mav0_dir / "cam0" / "data.csv"
        gt_csv = mav0_dir / "state_groundtruth_estimate0" / "data.csv"
        if cam0_csv.is_file() and gt_csv.is_file():
            sequence_dirs.append(root_path)

    return sorted(set(sequence_dirs))


def split_sequences(
    sequence_dirs: Sequence[Path], train_split_ratio: float
) -> Tuple[List[Path], List[Path]]:
    if not sequence_dirs:
        return [], []

    if len(sequence_dirs) == 1:
        return list(sequence_dirs), []

    split_idx = int(round(len(sequence_dirs) * train_split_ratio))
    split_idx = min(max(split_idx, 1), len(sequence_dirs) - 1)
    return list(sequence_dirs[:split_idx]), list(sequence_dirs[split_idx:])


def load_camera_sensor(
    sensor_yaml_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(sensor_yaml_path, "r", encoding="utf-8") as f:
        sensor_data = yaml.safe_load(f)

    fu, fv, cu, cv = sensor_data["intrinsics"]
    intrinsics = np.array(
        [
            [fu, 0.0, cu],
            [0.0, fv, cv],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    distortion = np.array(
        sensor_data.get("distortion_coefficients", [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float32,
    )
    body_from_camera = np.array(
        sensor_data["T_BS"]["data"], dtype=np.float32
    ).reshape(4, 4)
    return intrinsics, distortion, body_from_camera


def load_groundtruth(gt_csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    timestamps: List[int] = []
    poses: List[np.ndarray] = []

    with open(gt_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue

            timestamp = int(row[0])
            tx, ty, tz = [float(v) for v in row[1:4]]
            qw, qx, qy, qz = [float(v) for v in row[4:8]]

            world_from_body = np.eye(4, dtype=np.float32)
            world_from_body[:3, :3] = quat_wxyz_to_rotmat(qw, qx, qy, qz)
            world_from_body[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)

            timestamps.append(timestamp)
            poses.append(world_from_body)

    if not timestamps:
        raise ValueError(f"No valid ground-truth pose rows found in {gt_csv_path}")

    return np.asarray(timestamps, dtype=np.int64), np.stack(poses, axis=0)


def load_image_rows(image_csv_path: Path) -> List[Tuple[int, str]]:
    image_rows: List[Tuple[int, str]] = []

    with open(image_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            image_rows.append((int(row[0]), row[1]))

    return image_rows


def load_imu_data(imu_csv_path: Path) -> Dict[str, List]:
    timestamps: List[int] = []
    gyros: List[List[float]] = []
    accels: List[List[float]] = []

    with open(imu_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            timestamps.append(int(row[0]))
            gyros.append([float(v) for v in row[1:4]])
            accels.append([float(v) for v in row[4:7]])

    return {
        "timestamps_ns": timestamps,
        "gyro": gyros,
        "accel": accels,
    }


def quat_wxyz_to_rotmat(
    qw: float, qx: float, qy: float, qz: float
) -> np.ndarray:
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    quat_norm = np.linalg.norm(quat)
    if quat_norm < 1e-12:
        return np.eye(3, dtype=np.float32)

    qw, qx, qy, qz = quat / quat_norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def find_nearest_timestamp(
    timestamps: np.ndarray, query_timestamp: int
) -> Tuple[Optional[int], Optional[int]]:
    if len(timestamps) == 0:
        return None, None

    insert_idx = int(np.searchsorted(timestamps, query_timestamp))
    candidate_indices = []
    if insert_idx < len(timestamps):
        candidate_indices.append(insert_idx)
    if insert_idx > 0:
        candidate_indices.append(insert_idx - 1)

    best_index = None
    best_dt = None
    for candidate_idx in candidate_indices:
        candidate_dt = abs(int(timestamps[candidate_idx]) - int(query_timestamp))
        if best_dt is None or candidate_dt < best_dt:
            best_index = candidate_idx
            best_dt = candidate_dt

    return best_index, best_dt


def read_image_size(image_path: Path) -> Optional[List[int]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    return [int(image.shape[0]), int(image.shape[1])]


def compute_undistorted_intrinsics(
    intrinsics: np.ndarray, distortion: np.ndarray, image_size: Sequence[int]
) -> np.ndarray:
    height, width = int(image_size[0]), int(image_size[1])
    undistorted_intrinsics, _ = cv2.getOptimalNewCameraMatrix(
        intrinsics, distortion, (width, height), 0.0, (width, height)
    )
    return undistorted_intrinsics.astype(np.float32)


def build_camera_annotation(
    euroc_dir: Path,
    sequence_dir: Path,
    camera_name: str,
    gt_timestamps: np.ndarray,
    world_from_body: np.ndarray,
    imu_data: Optional[Dict[str, List]],
    max_pose_time_diff_ns: int,
    multi_camera: bool,
) -> Tuple[Optional[str], Optional[Dict], Dict[str, int]]:
    mav0_dir = sequence_dir / "mav0"
    camera_dir = mav0_dir / camera_name
    camera_csv = camera_dir / "data.csv"
    camera_sensor_yaml = camera_dir / "sensor.yaml"

    stats = {
        "total_images": 0,
        "matched_frames": 0,
        "missing_images": 0,
        "pose_gap_skipped": 0,
    }

    if not camera_csv.is_file() or not camera_sensor_yaml.is_file():
        stats["missing_images"] += 1
        return None, None, stats

    intrinsics, distortion, body_from_camera = load_camera_sensor(camera_sensor_yaml)
    image_rows = load_image_rows(camera_csv)

    frames = []
    image_size = None
    for image_timestamp, image_name in image_rows:
        stats["total_images"] += 1
        gt_index, pose_dt = find_nearest_timestamp(gt_timestamps, image_timestamp)
        if gt_index is None or pose_dt > max_pose_time_diff_ns:
            stats["pose_gap_skipped"] += 1
            continue

        image_path = camera_dir / "data" / image_name
        if not image_path.is_file():
            stats["missing_images"] += 1
            continue

        if image_size is None:
            image_size = read_image_size(image_path)
            if image_size is None:
                stats["missing_images"] += 1
                continue

        world_from_camera = world_from_body[gt_index] @ body_from_camera
        camera_from_world = np.linalg.inv(world_from_camera).astype(np.float32)[:3]

        frames.append(
            {
                "timestamp_ns": int(image_timestamp),
                "gt_timestamp_ns": int(gt_timestamps[gt_index]),
                "pose_dt_ns": int(pose_dt),
                "image_rel_path": image_path.relative_to(euroc_dir).as_posix(),
                "extrinsics": camera_from_world.tolist(),
            }
        )
        stats["matched_frames"] += 1

    if image_size is None:
        return None, None, stats

    sequence_name = sequence_dir.relative_to(euroc_dir).as_posix()
    if multi_camera:
        sequence_name = f"{sequence_name}:{camera_name}"

    sequence_payload = {
        "camera_name": camera_name,
        "sensor": {
            "intrinsics": intrinsics.tolist(),
            "distortion": distortion.tolist(),
            "undistorted_intrinsics": compute_undistorted_intrinsics(
                intrinsics, distortion, image_size
            ).tolist(),
            "image_size": image_size,
        },
        "frames": frames,
        "imu_data": imu_data,
    }
    return sequence_name, sequence_payload, stats


def build_split_annotations(
    euroc_dir: Path,
    sequence_dirs: Sequence[Path],
    camera_names: Sequence[str],
    max_pose_time_diff_ns: int,
) -> Tuple[Dict[str, Dict], Dict[str, int]]:
    outputs: Dict[str, Dict] = {}
    stats = {
        "sequences": 0,
        "camera_entries": 0,
        "matched_frames": 0,
        "missing_images": 0,
        "pose_gap_skipped": 0,
    }

    multi_camera = len(camera_names) > 1
    for sequence_dir in sequence_dirs:
        gt_timestamps, world_from_body = load_groundtruth(
            sequence_dir / "mav0" / "state_groundtruth_estimate0" / "data.csv"
        )
        imu_csv = sequence_dir / "mav0" / "imu0" / "data.csv"
        imu_data = load_imu_data(imu_csv) if imu_csv.is_file() else None

        stats["sequences"] += 1
        for camera_name in camera_names:
            sequence_name, payload, camera_stats = build_camera_annotation(
                euroc_dir=euroc_dir,
                sequence_dir=sequence_dir,
                camera_name=camera_name,
                gt_timestamps=gt_timestamps,
                world_from_body=world_from_body,
                imu_data=imu_data,
                max_pose_time_diff_ns=max_pose_time_diff_ns,
                multi_camera=multi_camera,
            )
            stats["matched_frames"] += camera_stats["matched_frames"]
            stats["missing_images"] += camera_stats["missing_images"]
            stats["pose_gap_skipped"] += camera_stats["pose_gap_skipped"]

            if sequence_name is None or payload is None:
                continue

            outputs[sequence_name] = payload
            stats["camera_entries"] += 1

    return outputs, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate local EuRoC annotations compatible with VGGT training."
    )
    parser.add_argument("--euroc_dir", required=True, help="Path to local EuRoC root.")
    parser.add_argument(
        "--output_dir", required=True, help="Directory to write generated *.jgz files."
    )
    parser.add_argument(
        "--camera_names",
        nargs="+",
        default=["cam0"],
        help="Camera names to export, e.g. cam0 cam1.",
    )
    parser.add_argument(
        "--train_split_ratio",
        type=float,
        default=0.8,
        help="Fraction of sorted sequences assigned to train split.",
    )
    parser.add_argument(
        "--max_pose_time_diff_ns",
        type=int,
        default=10_000_000,
        help="Maximum allowed image/GT timestamp mismatch in nanoseconds.",
    )
    args = parser.parse_args()

    euroc_dir = Path(args.euroc_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_dirs = discover_sequences(euroc_dir)
    train_sequences, test_sequences = split_sequences(
        sequence_dirs, args.train_split_ratio
    )

    train_payload, train_stats = build_split_annotations(
        euroc_dir=euroc_dir,
        sequence_dirs=train_sequences,
        camera_names=args.camera_names,
        max_pose_time_diff_ns=args.max_pose_time_diff_ns,
    )
    test_payload, test_stats = build_split_annotations(
        euroc_dir=euroc_dir,
        sequence_dirs=test_sequences,
        camera_names=args.camera_names,
        max_pose_time_diff_ns=args.max_pose_time_diff_ns,
    )

    dump_jgz(output_dir / "euroc_train.jgz", train_payload)
    dump_jgz(output_dir / "euroc_test.jgz", test_payload)

    summary = {
        "camera_names": list(args.camera_names),
        "train_split_ratio": args.train_split_ratio,
        "max_pose_time_diff_ns": args.max_pose_time_diff_ns,
        "sequence_dirs": [p.relative_to(euroc_dir).as_posix() for p in sequence_dirs],
        "train": train_stats,
        "test": test_stats,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Generated annotations under: {output_dir}")
    print(f"Discovered sequences: {len(sequence_dirs)}")
    print(f"Train camera entries: {train_stats['camera_entries']}")
    print(f"Test camera entries: {test_stats['camera_entries']}")
    print(
        "Matched frames:",
        train_stats["matched_frames"] + test_stats["matched_frames"],
    )


if __name__ == "__main__":
    main()
