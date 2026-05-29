#!/usr/bin/env python3
"""将 UMA VI 数据集重组为 EuRoC MAV (ASL) 格式。

UMA VI 每个序列有4个相机:
  - cam0/cam1: Bumblebee 立体 (1024x768, equidistant)
  - cam2/cam3: UEye 立体 (752x480, radtan)

转换后结构:
  <sequence>/
  └── mav0/
      ├── cam0/  (Bumblebee left)
      ├── cam1/  (Bumblebee right)
      ├── cam2/  (UEye left)
      ├── cam3/  (UEye right)
      ├── imu0/
      └── state_groundtruth_estimate0/
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import yaml


CAMERA_CALIB_MAP: Mapping[str, Tuple[str, str]] = {
    "cam0": ("camchain-imu-bumblebee.yaml", "cam0"),
    "cam1": ("camchain-imu-bumblebee.yaml", "cam1"),
    "cam2": ("camchain-imu-ueye.yaml", "cam0"),
    "cam3": ("camchain-imu-ueye.yaml", "cam1"),
}


def _format_inline_list(values: list, indent: int = 8, per_line: int = 4) -> str:
    parts = []
    for v in values:
        if isinstance(v, float):
            parts.append(f"{v!r}")
        else:
            parts.append(str(v))
    if len(parts) <= per_line:
        return "[" + ", ".join(parts) + "]"
    lines = []
    for i in range(0, len(parts), per_line):
        chunk = parts[i:i + per_line]
        prefix = " " * indent if i > 0 else ""
        lines.append(prefix + ", ".join(chunk))
    return "[" + ",\n".join(lines) + "]"


def _normalize_distortion_model(model: str) -> str:
    if model == "radtan":
        return "radial-tangential"
    return model


def _write_camera_sensor_yaml(path: Path, camera_name: str, camera_config: Dict,
                               rate_hz: float = 10.0) -> None:
    T_cam_imu = np.asarray(camera_config["T_cam_imu"], dtype=np.float64).reshape(4, 4)
    body_from_camera = np.linalg.inv(T_cam_imu)
    tbs_flat = body_from_camera.reshape(-1).tolist()

    intrinsics = list(camera_config["intrinsics"])
    dist_model = _normalize_distortion_model(camera_config.get("distortion_model", ""))
    dist_coeffs = list(camera_config.get("distortion_coeffs", [0.0, 0.0, 0.0, 0.0]))
    resolution = list(camera_config.get("resolution", [752, 480]))

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# General sensor definitions.\n")
        f.write("sensor_type: camera\n")
        f.write(f"comment: UMA-VI {camera_name}\n")
        f.write("\n")
        f.write("# Sensor extrinsics wrt. the body-frame.\n")
        f.write("T_BS:\n")
        f.write("  cols: 4\n")
        f.write("  rows: 4\n")
        f.write(f"  data: {_format_inline_list(tbs_flat)}\n")
        f.write("\n")
        f.write("# Camera specific definitions.\n")
        f.write(f"rate_hz: {rate_hz}\n")
        f.write(f"resolution: {_format_inline_list(resolution)}\n")
        f.write("camera_model: pinhole\n")
        f.write(f"intrinsics: {_format_inline_list(intrinsics)}  # fu, fv, cu, cv\n")
        f.write(f"distortion_model: {dist_model}\n")
        f.write(f"distortion_coefficients: {_format_inline_list(dist_coeffs)}\n")


def _write_imu_sensor_yaml(path: Path, imu_config: Dict) -> None:
    imu = imu_config.get("imu0", imu_config)
    identity = np.eye(4, dtype=float).reshape(-1).tolist()
    rate_hz = float(imu.get("update_rate", 250.0))

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Default imu sensor yaml file\n")
        f.write("sensor_type: imu\n")
        f.write("comment: UMA-VI IMU (Xsens MTi-100)\n")
        f.write("\n")
        f.write("# Sensor extrinsics wrt. the body-frame.\n")
        f.write("T_BS:\n")
        f.write("  cols: 4\n")
        f.write("  rows: 4\n")
        f.write(f"  data: {_format_inline_list(identity)}\n")
        f.write(f"rate_hz: {rate_hz}\n")
        f.write("\n")
        f.write("# inertial sensor noise model parameters (static)\n")
        f.write(f"gyroscope_noise_density: {imu.get('gyroscope_noise_density', 0.0)!r}\n")
        f.write(f"gyroscope_random_walk: {imu.get('gyroscope_random_walk', 0.0)!r}\n")
        f.write(f"accelerometer_noise_density: {imu.get('accelerometer_noise_density', 0.0)!r}\n")
        f.write(f"accelerometer_random_walk: {imu.get('accelerometer_random_walk', 0.0)!r}\n")


def _write_gt_sensor_yaml(path: Path) -> None:
    identity = np.eye(4, dtype=float).reshape(-1).tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# General sensor definitions.\n")
        f.write("sensor_type: visual-inertial\n")
        f.write("comment: UMA-VI ground truth (IMU trajectory)\n")
        f.write("\n")
        f.write("# Sensor extrinsics wrt. the body-frame.\n")
        f.write("T_BS:\n")
        f.write("  cols: 4\n")
        f.write("  rows: 4\n")
        f.write(f"  data: {_format_inline_list(identity)}\n")


def convert_image_csv(input_csv: Path, output_csv: Path) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with open(input_csv, "r", encoding="utf-8") as fin, \
         open(output_csv, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        writer.writerow(["#timestamp [ns]", "filename"])
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            writer.writerow([int(row[0]), row[1]])
            row_count += 1
    return row_count


def convert_gt_csv(input_csv: Path, output_csv: Path) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with open(input_csv, "r", encoding="utf-8") as fin, \
         open(output_csv, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        writer.writerow([
            "#timestamp",
            "p_RS_R_x [m]", "p_RS_R_y [m]", "p_RS_R_z [m]",
            "q_RS_w []", "q_RS_x []", "q_RS_y []", "q_RS_z []",
            "v_RS_R_x [m s^-1]", "v_RS_R_y [m s^-1]", "v_RS_R_z [m s^-1]",
            "b_w_RS_S_x [rad s^-1]", "b_w_RS_S_y [rad s^-1]", "b_w_RS_S_z [rad s^-1]",
            "b_a_RS_S_x [m s^-2]", "b_a_RS_S_y [m s^-2]", "b_a_RS_S_z [m s^-2]",
        ])
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            writer.writerow([
                int(row[0]),
                *[float(v) for v in row[1:8]],
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ])
            row_count += 1
    return row_count


def discover_uma_vi_sequences(uma_vi_dir: Path) -> List[Path]:
    sequences = []
    for entry in sorted(uma_vi_dir.iterdir()):
        if not entry.is_dir() or entry.name == "calibration":
            continue
        if (entry / "cam0" / "data.csv").is_file():
            sequences.append(entry)
    return sequences


def process_sequence(
    sequence_dir: Path,
    calib_store: Dict[str, Dict],
    imu_yaml_path: Path,
    dry_run: bool = False,
) -> Dict:
    mav0 = sequence_dir / "mav0"
    ops = []
    stats = {"cameras": 0, "gt_frames": 0, "errors": []}

    # 各相机组的典型帧率
    rate_hz_map = {"cam0": 10.0, "cam1": 10.0, "cam2": 10.0, "cam3": 10.0}

    for camera_name in ("cam0", "cam1", "cam2", "cam3"):
        src_cam_dir = sequence_dir / camera_name
        dst_cam_dir = mav0 / camera_name
        if not src_cam_dir.is_dir():
            continue

        calib_file_name, entry_name = CAMERA_CALIB_MAP.get(camera_name, (None, None))
        if calib_file_name is None:
            stats["errors"].append(f"未找到 {camera_name} 标定映射")
            continue

        camera_calib = None
        for path_str, content in calib_store.items():
            if Path(path_str).name == calib_file_name:
                camera_calib = content.get(entry_name)
                break
        if camera_calib is None:
            stats["errors"].append(f"未找到 {camera_name} 的标定数据")
            continue

        if not dry_run:
            _write_camera_sensor_yaml(
                dst_cam_dir / "sensor.yaml", camera_name, camera_calib,
                rate_hz=rate_hz_map.get(camera_name, 10.0),
            )
        ops.append(f"  write {dst_cam_dir.relative_to(sequence_dir)}/sensor.yaml")

        csv_count = 0
        if not dry_run:
            csv_count = convert_image_csv(src_cam_dir / "data.csv", dst_cam_dir / "data.csv")
        ops.append(f"  convert data.csv ({csv_count} rows)")

        src_data = src_cam_dir / "data"
        dst_data = dst_cam_dir / "data"
        if src_data.is_dir():
            if not dry_run:
                if dst_data.is_symlink() or dst_data.exists():
                    dst_data.unlink()
                dst_data.symlink_to(os.path.relpath(src_data, dst_data.parent))
            ops.append(f"  symlink data/ -> {os.path.relpath(src_data, dst_data.parent)}")

        stats["cameras"] += 1

    # IMU
    src_imu = sequence_dir / "imu0"
    dst_imu = mav0 / "imu0"
    if src_imu.is_dir():
        imu_config = {}
        if imu_yaml_path.is_file():
            with open(imu_yaml_path, "r", encoding="utf-8") as f:
                imu_config = yaml.safe_load(f)
        if not dry_run:
            _write_imu_sensor_yaml(dst_imu / "sensor.yaml", imu_config)
        ops.append(f"  write {dst_imu.relative_to(sequence_dir)}/sensor.yaml")

        src_imu_csv = src_imu / "data.csv"
        dst_imu_csv = dst_imu / "data.csv"
        if src_imu_csv.is_file() and not dry_run:
            if dst_imu_csv.is_symlink() or dst_imu_csv.exists():
                dst_imu_csv.unlink()
            dst_imu_csv.symlink_to(os.path.relpath(src_imu_csv, dst_imu_csv.parent))
        ops.append(f"  symlink imu0/data.csv")

    # GT
    gt_dir = mav0 / "state_groundtruth_estimate0"
    traj_csv = sequence_dir / "imu0_trajectory.csv"
    n_frames = 0
    if traj_csv.is_file():
        if not dry_run:
            n_frames = convert_gt_csv(traj_csv, gt_dir / "data.csv")
            _write_gt_sensor_yaml(gt_dir / "sensor.yaml")
        ops.append(f"  write GT data.csv ({n_frames} frames)")
        stats["gt_frames"] = n_frames

    return {"ops": ops, "stats": stats}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 UMA VI 数据集重组为 EuRoC MAV (ASL) 格式"
    )
    parser.add_argument("--uma_vi_dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    uma_dir = Path(args.uma_vi_dir).resolve()
    calib_dir = uma_dir / "calibration"
    if not calib_dir.is_dir():
        print(f"错误：标定目录不存在: {calib_dir}")
        return

    sequences = discover_uma_vi_sequences(uma_dir)
    if not sequences:
        print(f"在 {uma_dir} 中未发现 UMA VI 序列")
        return

    needs_unzip = []
    for seq_dir in sequences:
        cam0_data = seq_dir / "cam0" / "data"
        if not cam0_data.is_dir() or not list(cam0_data.glob("*.png")):
            needs_unzip.append(seq_dir.name)
    if needs_unzip:
        print("以下序列缺少解压后的图片数据，请先解压同级 .zip 文件:")
        for name in needs_unzip:
            print(f"  {name}")
        return

    calib_store = {}
    for fname in ("camchain-imu-bumblebee.yaml", "camchain-imu-ueye.yaml"):
        cf = calib_dir / fname
        if cf.is_file():
            with open(cf, "r", encoding="utf-8") as fh:
                calib_store[str(cf)] = yaml.safe_load(fh)
    imu_yaml_path = calib_dir / "imu-xsens.yaml"

    print(f"发现 {len(sequences)} 个序列")
    if args.dry_run:
        print("(dry-run 模式)\n")

    total = {"cameras": 0, "gt_frames": 0, "errors": 0}
    for seq_dir in sequences:
        print(f"\n处理: {seq_dir.name}")
        result = process_sequence(seq_dir, calib_store, imu_yaml_path, dry_run=args.dry_run)
        for op in result["ops"]:
            print(op)
        total["cameras"] += result["stats"]["cameras"]
        total["gt_frames"] += result["stats"]["gt_frames"]
        total["errors"] += len(result["stats"]["errors"])

    print(f"\n===== 汇总 =====")
    print(f"序列数: {len(sequences)}")
    print(f"相机条目数: {total['cameras']}")
    print(f"GT frame 总数: {total['gt_frames']}")
    if args.dry_run:
        print("\n(dry-run — 无实际写入)")


if __name__ == "__main__":
    main()
