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

try:
    from training.data.preprocess.vi_schema import (
        add_clean_degradation_defaults,
        attach_sequence_metadata,
        ensure_frame_extrinsics_aliases,
        validate_vi_annotation,
    )
except ImportError:
    from vi_schema import (
        add_clean_degradation_defaults,
        attach_sequence_metadata,
        ensure_frame_extrinsics_aliases,
        validate_vi_annotation,
    )


def dump_jgz(path: Path, payload: Dict) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)


def discover_sequences(tum_vi_dir: Path) -> List[Path]:
    sequence_dirs: List[Path] = []

    for root, dirs, _ in os.walk(tum_vi_dir, followlinks=True):
        dirs[:] = [d for d in dirs if d != "__MACOSX"]
        root_path = Path(root)
        if (
            (root_path / "mav0" / "cam0" / "data.csv").is_file()
            and (root_path / "mav0" / "mocap0" / "data.csv").is_file()
            and (root_path / "dso" / "camchain.yaml").is_file()
        ):
            sequence_dirs.append(root_path)

    return sorted(set(sequence_dirs))


def make_intrinsics_matrix(intrinsics_values: Sequence[float]) -> np.ndarray:
    fx, fy, cx, cy = intrinsics_values
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def load_camchain(camchain_path: Path) -> Dict[str, Dict]:
    with open(camchain_path, "r", encoding="utf-8") as f:
        camchain = yaml.safe_load(f)

    cameras = {}
    for camera_name, camera_data in camchain.items():
        if not camera_name.startswith("cam"):
            continue

        intrinsics = make_intrinsics_matrix(camera_data["intrinsics"])
        distortion = np.asarray(
            camera_data.get("distortion_coeffs", [0.0, 0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        camera_from_imu = np.asarray(
            camera_data["T_cam_imu"], dtype=np.float32
        ).reshape(4, 4)
        resolution = camera_data.get("resolution")
        image_size = None
        if resolution is not None:
            width, height = [int(v) for v in resolution]
            image_size = [height, width]

        cameras[camera_name] = {
            "intrinsics": intrinsics,
            "distortion": distortion,
            "distortion_model": camera_data.get("distortion_model", ""),
            "camera_model": camera_data.get("camera_model", ""),
            "camera_from_imu": camera_from_imu,
            "image_size": image_size,
        }

    return cameras


def load_mocap_groundtruth(mocap_csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    timestamps: List[int] = []
    poses: List[np.ndarray] = []

    with open(mocap_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue

            timestamp = int(row[0])
            tx, ty, tz = [float(v) for v in row[1:4]]
            qw, qx, qy, qz = [float(v) for v in row[4:8]]

            world_from_imu = np.eye(4, dtype=np.float32)
            world_from_imu[:3, :3] = quat_wxyz_to_rotmat(qw, qx, qy, qz)
            world_from_imu[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)

            timestamps.append(timestamp)
            poses.append(world_from_imu)

    if not timestamps:
        raise ValueError(f"No valid mocap pose rows found in {mocap_csv_path}")

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
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    image_size: Sequence[int],
    distortion_model: str,
) -> np.ndarray:
    height, width = int(image_size[0]), int(image_size[1])
    if distortion_model == "equidistant":
        return cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            intrinsics,
            distortion.reshape(-1, 1),
            (width, height),
            np.eye(3),
            balance=0.0,
        ).astype(np.float32)

    undistorted_intrinsics, _ = cv2.getOptimalNewCameraMatrix(
        intrinsics, distortion, (width, height), 0.0, (width, height)
    )
    return undistorted_intrinsics.astype(np.float32)


def compute_extrinsics_w2c(
    world_from_imu: np.ndarray, camera_from_imu: np.ndarray
) -> np.ndarray:
    imu_from_camera = np.linalg.inv(camera_from_imu)
    world_from_camera = world_from_imu @ imu_from_camera
    camera_from_world = np.linalg.inv(world_from_camera)
    return camera_from_world.astype(np.float32)[:3]


def build_camera_annotation(
    tum_vi_dir: Path,
    sequence_dir: Path,
    camera_name: str,
    split: str,
    camera_calibration: Dict,
    mocap_timestamps: np.ndarray,
    world_from_imu: np.ndarray,
    imu_data: Optional[Dict[str, List]],
    max_pose_time_diff_ns: int,
    multi_camera: bool,
) -> Tuple[Optional[str], Optional[Dict], Dict[str, int]]:
    camera_dir = sequence_dir / "mav0" / camera_name
    camera_csv = camera_dir / "data.csv"

    stats = {
        "total_images": 0,
        "matched_frames": 0,
        "missing_images": 0,
        "pose_gap_skipped": 0,
    }

    if not camera_csv.is_file():
        stats["missing_images"] += 1
        return None, None, stats

    image_rows = load_image_rows(camera_csv)
    image_size = camera_calibration.get("image_size")
    frames = []
    for image_timestamp, image_name in image_rows:
        stats["total_images"] += 1
        gt_index, pose_dt = find_nearest_timestamp(mocap_timestamps, image_timestamp)
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

        extrinsics_w2c = compute_extrinsics_w2c(
            world_from_imu[gt_index],
            camera_calibration["camera_from_imu"],
        )

        frames.append(
            {
                "frame_id": len(frames),
                "timestamp_ns": int(image_timestamp),
                "gt_timestamp_ns": int(mocap_timestamps[gt_index]),
                "pose_dt_ns": int(pose_dt),
                "image_rel_path": image_path.relative_to(tum_vi_dir).as_posix(),
                "clean_image_rel_path": image_path.relative_to(tum_vi_dir).as_posix(),
                "extrinsics": extrinsics_w2c.tolist(),
                "extrinsics_w2c": extrinsics_w2c.tolist(),
            }
        )
        stats["matched_frames"] += 1

    if image_size is None:
        return None, None, stats

    sequence_name = sequence_dir.relative_to(tum_vi_dir).as_posix()
    if multi_camera:
        sequence_name = f"{sequence_name}:{camera_name}"

    intrinsics = camera_calibration["intrinsics"]
    distortion = camera_calibration["distortion"]
    distortion_model = camera_calibration["distortion_model"]
    imu_from_camera = np.linalg.inv(camera_calibration["camera_from_imu"]).astype(
        np.float32
    )
    sequence_payload = {
        "camera_name": camera_name,
        "sensor": {
            "camera_model": camera_calibration["camera_model"],
            "distortion_model": distortion_model,
            "intrinsics": intrinsics.tolist(),
            "distortion": distortion.tolist(),
            "undistorted_intrinsics": compute_undistorted_intrinsics(
                intrinsics, distortion, image_size, distortion_model
            ).tolist(),
            "image_size": image_size,
            "T_cam_imu": camera_calibration["camera_from_imu"].tolist(),
            "T_imu_cam": imu_from_camera.tolist(),
            "camera_from_imu": camera_calibration["camera_from_imu"].tolist(),
        },
        "frames": frames,
        "imu_data": imu_data,
        "diagnostics": {
            "max_pose_time_diff_ns": int(max_pose_time_diff_ns),
            "num_frames": int(len(frames)),
            "num_missing_images": int(stats["missing_images"]),
            "num_missing_gt": int(stats["pose_gap_skipped"]),
        },
    }
    add_clean_degradation_defaults(sequence_payload["frames"])
    ensure_frame_extrinsics_aliases(sequence_payload["frames"])
    attach_sequence_metadata(
        payload=sequence_payload,
        dataset="tum_vi",
        sequence_name=sequence_name,
        sequence_path=sequence_name,
        camera_name=camera_name,
        split=split,
    )
    return sequence_name, sequence_payload, stats


def build_sequence_annotations(
    tum_vi_dir: Path,
    sequence_dir: Path,
    camera_names: Sequence[str],
    max_pose_time_diff_ns: int,
    split: str,
) -> Tuple[Dict[str, Dict], Dict[str, int]]:
    outputs: Dict[str, Dict] = {}
    stats = {
        "sequences": 1,
        "camera_entries": 0,
        "matched_frames": 0,
        "missing_images": 0,
        "pose_gap_skipped": 0,
    }

    cameras = load_camchain(sequence_dir / "dso" / "camchain.yaml")
    mocap_timestamps, world_from_imu = load_mocap_groundtruth(
        sequence_dir / "mav0" / "mocap0" / "data.csv"
    )
    imu_csv = sequence_dir / "mav0" / "imu0" / "data.csv"
    imu_data = load_imu_data(imu_csv) if imu_csv.is_file() else None

    multi_camera = len(camera_names) > 1
    for camera_name in camera_names:
        if camera_name not in cameras:
            stats["missing_images"] += 1
            continue

        sequence_name, payload, camera_stats = build_camera_annotation(
            tum_vi_dir=tum_vi_dir,
            sequence_dir=sequence_dir,
            camera_name=camera_name,
            split=split,
            camera_calibration=cameras[camera_name],
            mocap_timestamps=mocap_timestamps,
            world_from_imu=world_from_imu,
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


def sequence_output_stem(tum_vi_dir: Path, sequence_dir: Path) -> str:
    return sequence_dir.relative_to(tum_vi_dir).as_posix().replace("/", "__")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate local TUM-VI annotations compatible with VGGT training."
    )
    parser.add_argument("--tum_vi_dir", required=True, help="Path to local TUM-VI root.")
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
        "--max_pose_time_diff_ns",
        type=int,
        default=10_000_000,
        help="Maximum allowed image/mocap timestamp mismatch in nanoseconds.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Split label to write into generated TUM-VI annotations.",
    )
    args = parser.parse_args()

    tum_vi_dir = Path(args.tum_vi_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_dirs = discover_sequences(tum_vi_dir)
    total_stats = {
        "sequences": 0,
        "camera_entries": 0,
        "matched_frames": 0,
        "missing_images": 0,
        "pose_gap_skipped": 0,
    }
    generated_files = []
    per_sequence_stats = {}
    for sequence_dir in sequence_dirs:
        sequence_payload, sequence_stats = build_sequence_annotations(
            tum_vi_dir=tum_vi_dir,
            sequence_dir=sequence_dir,
            camera_names=args.camera_names,
            max_pose_time_diff_ns=args.max_pose_time_diff_ns,
            split=args.split,
        )
        validate_vi_annotation(sequence_payload)
        output_name = f"{sequence_output_stem(tum_vi_dir, sequence_dir)}.jgz"
        dump_jgz(output_dir / output_name, sequence_payload)
        generated_files.append(output_name)

        sequence_key = sequence_dir.relative_to(tum_vi_dir).as_posix()
        per_sequence_stats[sequence_key] = {
            "file": output_name,
            "stats": sequence_stats,
        }
        for key in total_stats:
            total_stats[key] += sequence_stats[key]

    summary = {
        "dataset_format": "tum_vi",
        "camera_names": list(args.camera_names),
        "max_pose_time_diff_ns": args.max_pose_time_diff_ns,
        "sequence_dirs": [p.relative_to(tum_vi_dir).as_posix() for p in sequence_dirs],
        "generated_files": generated_files,
        "total": total_stats,
        "per_sequence": per_sequence_stats,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Generated annotations under: {output_dir}")
    print(f"Discovered sequences: {len(sequence_dirs)}")
    print(f"Generated sequence files: {len(generated_files)}")
    print(f"Camera entries: {total_stats['camera_entries']}")
    print(f"Matched frames: {total_stats['matched_frames']}")


if __name__ == "__main__":
    main()
