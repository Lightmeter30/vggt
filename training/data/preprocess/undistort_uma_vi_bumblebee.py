#!/usr/bin/env python3
"""将 UMA-VI Bumblebee 鱼眼图像离线去畸变为普通 pinhole 图像。"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path
from typing import Iterable, Mapping

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
ZERO_DISTORTION = [0.0, 0.0, 0.0, 0.0]


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, payload: Mapping) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(payload), f, sort_keys=False)


def _copy_once(src: Path, dst: Path) -> bool:
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _backup_metadata_file(uma_vi_dir: Path, path: Path, origin_dir_name: str) -> None:
    backup_path = uma_vi_dir / origin_dir_name / "metadata" / path.relative_to(uma_vi_dir)
    _copy_once(path, backup_path)


def _discover_sequences(uma_vi_dir: Path) -> list[Path]:
    ignored = {"calibration", "anno", "origin", "original"}
    sequences = []
    for entry in sorted(uma_vi_dir.iterdir()):
        if entry.name in ignored or not entry.is_dir():
            continue
        if (entry / "mav0").is_dir():
            sequences.append(entry)
    return sequences


def _camera_matrix_from_sensor(sensor: Mapping) -> np.ndarray:
    intrinsics = sensor["intrinsics"]
    if len(intrinsics) == 4:
        fx, fy, cx, cy = [float(value) for value in intrinsics]
        return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)


def _intrinsics_list_from_matrix(matrix: np.ndarray) -> list[float]:
    return [
        float(matrix[0, 0]),
        float(matrix[1, 1]),
        float(matrix[0, 2]),
        float(matrix[1, 2]),
    ]


def _build_undistort_maps(
    sensor: Mapping,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = _camera_matrix_from_sensor(sensor)
    d = np.asarray(sensor.get("distortion_coefficients", ZERO_DISTORTION), dtype=np.float64).reshape(4, 1)
    new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        k,
        d,
        (width, height),
        np.eye(3, dtype=np.float64),
        balance=0.0,
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        k,
        d,
        np.eye(3, dtype=np.float64),
        new_k,
        (width, height),
        cv2.CV_16SC2,
    )
    return map1, map2, new_k


def _undistort_image(
    image: np.ndarray,
    sensor: Mapping,
    maps: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    if maps is None:
        maps = _build_undistort_maps(sensor, width, height)
    map1, map2, new_k = maps
    undistorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted, new_k


def _iter_images(data_dir: Path) -> Iterable[Path]:
    for path in sorted(data_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"写入去畸变图像失败: {path}")


def _read_image(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _ensure_readable_origin_image(source_path: Path, backup_path: Path) -> tuple[np.ndarray, bool]:
    repaired = False
    if backup_path.exists():
        image = _read_image(backup_path)
        if image is not None:
            return image, repaired
        backup_path.unlink()
        repaired = True

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, backup_path)
    image = _read_image(backup_path)
    if image is None:
        raise FileNotFoundError(f"读取图像失败: {backup_path}")
    return image, repaired


def _load_processing_sensor(
    uma_vi_dir: Path,
    sensor_path: Path,
    origin_dir_name: str,
) -> dict:
    metadata_backup = uma_vi_dir / origin_dir_name / "metadata" / sensor_path.relative_to(uma_vi_dir)
    if metadata_backup.is_file():
        return _load_yaml(metadata_backup)
    return _load_yaml(sensor_path)


def _update_sensor_yaml(
    *,
    uma_vi_dir: Path,
    sensor_path: Path,
    new_intrinsics: np.ndarray,
    origin_dir_name: str,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    _backup_metadata_file(uma_vi_dir, sensor_path, origin_dir_name)
    sensor = _load_yaml(sensor_path)
    sensor["intrinsics"] = _intrinsics_list_from_matrix(new_intrinsics)
    sensor["distortion_model"] = "radial-tangential"
    sensor["distortion_coefficients"] = list(ZERO_DISTORTION)
    _write_yaml(sensor_path, sensor)


def _update_annotation_payload(payload: dict, new_intrinsics_by_camera: Mapping[str, np.ndarray]) -> bool:
    changed = False
    for sequence in payload.values():
        camera_name = sequence.get("camera_name")
        if camera_name not in new_intrinsics_by_camera:
            continue
        sensor = sequence.get("sensor")
        if not isinstance(sensor, dict):
            continue
        new_k = new_intrinsics_by_camera[camera_name]
        sensor["intrinsics"] = new_k.astype(float).tolist()
        sensor["undistorted_intrinsics"] = new_k.astype(float).tolist()
        sensor["distortion"] = list(ZERO_DISTORTION)
        sensor["distortion_model"] = "radial-tangential"
        changed = True
    return changed


def _update_manifest_distortion_models(payload: dict, cameras: Iterable[str]) -> bool:
    changed = False
    for record in payload.get("sequences", {}).values():
        distortion_models = record.get("distortion_models")
        if not isinstance(distortion_models, dict):
            continue
        for camera_name in cameras:
            if distortion_models.get(camera_name) == "equidistant":
                distortion_models[camera_name] = "radial-tangential"
                changed = True
    nested = payload.get("sequence_manifest")
    if isinstance(nested, dict):
        changed = _update_manifest_distortion_models(nested, cameras) or changed
    return changed


def _update_json_manifest(path: Path, cameras: Iterable[str], dry_run: bool) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    changed = _update_manifest_distortion_models(payload, cameras)
    if changed and not dry_run:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return changed


def _update_annotations(
    *,
    uma_vi_dir: Path,
    new_intrinsics_by_camera: Mapping[str, np.ndarray],
    origin_dir_name: str,
    dry_run: bool,
) -> int:
    anno_dir = uma_vi_dir / "anno"
    if not anno_dir.is_dir():
        return 0

    changed_files = 0
    cameras = tuple(new_intrinsics_by_camera)
    for path in sorted(anno_dir.glob("*.jgz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        if not _update_annotation_payload(payload, new_intrinsics_by_camera):
            continue
        changed_files += 1
        if not dry_run:
            _backup_metadata_file(uma_vi_dir, path, origin_dir_name)
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(payload, f)

    for name in ("sequence_manifest.json", "summary.json"):
        path = anno_dir / name
        if not path.is_file():
            continue
        needs_update = _update_json_manifest(path, cameras, dry_run=True)
        if needs_update:
            changed_files += 1
            if not dry_run:
                _backup_metadata_file(uma_vi_dir, path, origin_dir_name)
                _update_json_manifest(path, cameras, dry_run=False)

    return changed_files


def process_uma_vi(
    uma_vi_dir: str | Path,
    *,
    cameras: Iterable[str] = ("cam0", "cam1"),
    origin_dir_name: str = "origin",
    dry_run: bool = False,
) -> dict:
    uma_vi_dir = Path(uma_vi_dir).resolve()
    if not uma_vi_dir.is_dir():
        raise FileNotFoundError(f"UMA-VI 目录不存在: {uma_vi_dir}")

    cameras = tuple(cameras)
    stats = {
        "sequences": 0,
        "processed_cameras": 0,
        "skipped_cameras": 0,
        "processed_images": 0,
        "planned_images": 0,
        "metadata_files": 0,
        "repaired_backups": 0,
    }
    last_intrinsics_by_camera: dict[str, np.ndarray] = {}

    for sequence_dir in _discover_sequences(uma_vi_dir):
        stats["sequences"] += 1
        for camera_name in cameras:
            sensor_path = sequence_dir / "mav0" / camera_name / "sensor.yaml"
            data_dir = sequence_dir / "mav0" / camera_name / "data"
            if not sensor_path.is_file() or not data_dir.exists():
                continue

            processing_sensor = _load_processing_sensor(uma_vi_dir, sensor_path, origin_dir_name)
            if processing_sensor.get("distortion_model") != "equidistant":
                stats["skipped_cameras"] += 1
                continue

            image_paths = list(_iter_images(data_dir))
            if not image_paths:
                continue
            stats["processed_cameras"] += 1
            stats["planned_images"] += len(image_paths)
            if dry_run:
                continue

            new_intrinsics = None
            maps_by_size = {}
            for image_path in image_paths:
                backup_path = uma_vi_dir / origin_dir_name / sequence_dir.name / camera_name / "data" / image_path.name
                image, repaired = _ensure_readable_origin_image(image_path, backup_path)
                if repaired:
                    stats["repaired_backups"] += 1
                height, width = image.shape[:2]
                maps = maps_by_size.get((width, height))
                if maps is None:
                    maps = _build_undistort_maps(processing_sensor, width, height)
                    maps_by_size[(width, height)] = maps
                undistorted, new_intrinsics = _undistort_image(image, processing_sensor, maps)
                _write_image(image_path, undistorted)
                stats["processed_images"] += 1

            if new_intrinsics is not None:
                last_intrinsics_by_camera[camera_name] = new_intrinsics
                _update_sensor_yaml(
                    uma_vi_dir=uma_vi_dir,
                    sensor_path=sensor_path,
                    new_intrinsics=new_intrinsics,
                    origin_dir_name=origin_dir_name,
                    dry_run=dry_run,
                )

    if last_intrinsics_by_camera:
        stats["metadata_files"] = _update_annotations(
            uma_vi_dir=uma_vi_dir,
            new_intrinsics_by_camera=last_intrinsics_by_camera,
            origin_dir_name=origin_dir_name,
            dry_run=dry_run,
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="离线去除 UMA-VI cam0/cam1 Bumblebee 鱼眼畸变")
    parser.add_argument("--uma_vi_dir", default="dataset/UMA_VI")
    parser.add_argument("--cameras", nargs="+", default=["cam0", "cam1"])
    parser.add_argument("--origin_dir_name", default="origin")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stats = process_uma_vi(
        args.uma_vi_dir,
        cameras=args.cameras,
        origin_dir_name=args.origin_dir_name,
        dry_run=args.dry_run,
    )
    mode = "dry-run" if args.dry_run else "write"
    print(f"mode: {mode}")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
