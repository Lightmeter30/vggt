#!/usr/bin/env python3
"""将 TUM-VI rectified 目录原地补全为 EuRoC MAV (ASL) 格式。

TUM rectified 已具备大部分 ASL 结构，只需补全:
1. mav0/cam0/sensor.yaml + mav0/cam1/sensor.yaml (从 dso/camchain.yaml)
2. mav0/imu0/sensor.yaml (从 dso/imu_config.yaml)
3. mav0/state_groundtruth_estimate0/ (从 mav0/mocap0/data.csv)
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


def _format_inline_list(values: list, indent: int = 8, per_line: int = 4) -> str:
    """格式化为 YAML 内联数组，每行 per_line 个值，续行对齐."""
    parts = []
    for i, v in enumerate(values):
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


def _write_camera_sensor_yaml(path: Path, camera_name: str, camera_config: Dict,
                               rate_hz: float = 20.0) -> None:
    """写 ASL 格式 camera sensor.yaml，对齐 EuRoC 风格."""
    T_cam_imu = np.asarray(camera_config["T_cam_imu"], dtype=np.float64).reshape(4, 4)
    body_from_camera = np.linalg.inv(T_cam_imu)
    tbs_flat = body_from_camera.reshape(-1).tolist()

    intrinsics = list(camera_config["intrinsics"])
    # 已校正图像使用 zero distortion
    dist_coeffs = [0.0, 0.0, 0.0, 0.0]
    resolution = list(camera_config.get("resolution", [512, 512]))

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# General sensor definitions.\n")
        f.write("sensor_type: camera\n")
        f.write(f"comment: TUM-VI {camera_name} (rectified)\n")
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
        f.write("distortion_model: radial-tangential\n")
        f.write(f"distortion_coefficients: {_format_inline_list(dist_coeffs)}\n")


def _write_imu_sensor_yaml(path: Path, imu_config: Optional[Dict]) -> None:
    """写 ASL 格式 IMU sensor.yaml，对齐 EuRoC 风格."""
    rate_hz = 200.0
    gyro_noise = None
    gyro_rw = None
    accel_noise = None
    accel_rw = None
    comment = "TUM-VI IMU (rectified)"

    if imu_config:
        rate_hz = float(imu_config.get("update_rate", 200.0))
        gyro_noise = imu_config.get("gyroscope_noise_density")
        gyro_rw = imu_config.get("gyroscope_random_walk")
        accel_noise = imu_config.get("accelerometer_noise_density")
        accel_rw = imu_config.get("accelerometer_random_walk")

    identity = np.eye(4, dtype=float).reshape(-1).tolist()

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Default imu sensor yaml file\n")
        f.write("sensor_type: imu\n")
        f.write(f"comment: {comment}\n")
        f.write("\n")
        f.write("# Sensor extrinsics wrt. the body-frame.\n")
        f.write("T_BS:\n")
        f.write("  cols: 4\n")
        f.write("  rows: 4\n")
        f.write(f"  data: {_format_inline_list(identity)}\n")
        f.write(f"rate_hz: {rate_hz}\n")
        f.write("\n")
        f.write("# inertial sensor noise model parameters (static)\n")
        if gyro_noise is not None:
            f.write(f"gyroscope_noise_density: {gyro_noise!r}\n")
        if gyro_rw is not None:
            f.write(f"gyroscope_random_walk: {gyro_rw!r}\n")
        if accel_noise is not None:
            f.write(f"accelerometer_noise_density: {accel_noise!r}\n")
        if accel_rw is not None:
            f.write(f"accelerometer_random_walk: {accel_rw!r}\n")


def _write_gt_sensor_yaml(path: Path) -> None:
    """写 ASL 格式 state_groundtruth_estimate0 sensor.yaml."""
    identity = np.eye(4, dtype=float).reshape(-1).tolist()

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# General sensor definitions.\n")
        f.write("sensor_type: visual-inertial\n")
        f.write("comment: TUM-VI ground truth (mocap)\n")
        f.write("\n")
        f.write("# Sensor extrinsics wrt. the body-frame.\n")
        f.write("T_BS:\n")
        f.write("  cols: 4\n")
        f.write("  rows: 4\n")
        f.write(f"  data: {_format_inline_list(identity)}\n")


def convert_gt_csv(input_csv: Path, output_csv: Path) -> int:
    """将 mocap0 GT CSV 转换为 ASL state_groundtruth_estimate0 格式.

    原格式: #timestamp [ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z
    目标格式: #timestamp, p_x, p_y, p_z, q_w, q_x, q_y, q_z, v(3x0), bias(6x0)
    """
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
                0.0, 0.0, 0.0,  # v
                0.0, 0.0, 0.0,  # bw
                0.0, 0.0, 0.0,  # ba
            ])
            row_count += 1
    return row_count


def discover_tum_rectified_sequences(tum_rectified_dir: Path) -> List[Path]:
    sequence_dirs = []
    for root, dirs, _ in os.walk(tum_rectified_dir, followlinks=True):
        dirs[:] = [d for d in dirs if d != "__MACOSX"]
        root_path = Path(root)
        if ((root_path / "mav0" / "cam0" / "data.csv").is_file()
                and (root_path / "mav0" / "mocap0" / "data.csv").is_file()
                and (root_path / "dso" / "camchain.yaml").is_file()):
            sequence_dirs.append(root_path)
    return sorted(set(sequence_dirs))


def process_sequence(sequence_dir: Path, dry_run: bool = False) -> Dict:
    mav0 = sequence_dir / "mav0"
    dso = sequence_dir / "dso"

    with open(dso / "camchain.yaml", "r", encoding="utf-8") as f:
        camchain = yaml.safe_load(f)

    imu_config = None
    imu_config_path = dso / "imu_config.yaml"
    if imu_config_path.is_file():
        with open(imu_config_path, "r", encoding="utf-8") as f:
            imu_config = yaml.safe_load(f)

    ops = []
    stats = {"sensor_yaml_count": 0, "gt_frames": 0}

    # 1. 生成 camera sensor.yaml
    for camera_name in ("cam0", "cam1"):
        if camera_name not in camchain:
            continue
        sensor_path = mav0 / camera_name / "sensor.yaml"
        if not dry_run:
            _write_camera_sensor_yaml(sensor_path, camera_name, camchain[camera_name])
        ops.append(f"  write {sensor_path.relative_to(sequence_dir)}")
        stats["sensor_yaml_count"] += 1

    # 2. 生成 IMU sensor.yaml
    imu_sensor_path = mav0 / "imu0" / "sensor.yaml"
    if not dry_run:
        _write_imu_sensor_yaml(imu_sensor_path, imu_config)
    ops.append(f"  write {imu_sensor_path.relative_to(sequence_dir)}")
    stats["sensor_yaml_count"] += 1

    # 3. 生成 state_groundtruth_estimate0
    gt_dir = mav0 / "state_groundtruth_estimate0"
    mocap_csv = mav0 / "mocap0" / "data.csv"
    gt_csv = gt_dir / "data.csv"

    n_frames = 0
    if not dry_run:
        n_frames = convert_gt_csv(mocap_csv, gt_csv)
        _write_gt_sensor_yaml(gt_dir / "sensor.yaml")
    ops.append(f"  write {gt_csv.relative_to(sequence_dir)} ({n_frames} frames)")
    stats["gt_frames"] = n_frames

    return {"ops": ops, "stats": stats}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 TUM-VI rectified 目录原地补全为 EuRoC MAV (ASL) 格式"
    )
    parser.add_argument("--tum_rectified_dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tum_dir = Path(args.tum_rectified_dir).resolve()
    sequences = discover_tum_rectified_sequences(tum_dir)
    if not sequences:
        print(f"在 {tum_dir} 中未发现 TUM rectified 序列")
        return

    print(f"发现 {len(sequences)} 个序列")
    if args.dry_run:
        print("(dry-run 模式，不实际写入)\n")

    total = {"sensor_yaml_count": 0, "gt_frames": 0}
    for seq_dir in sequences:
        rel = seq_dir.relative_to(tum_dir)
        print(f"\n处理: {rel}")
        result = process_sequence(seq_dir, dry_run=args.dry_run)
        for op in result["ops"]:
            print(op)
        total["sensor_yaml_count"] += result["stats"]["sensor_yaml_count"]
        total["gt_frames"] += result["stats"]["gt_frames"]

    print(f"\n===== 汇总 =====")
    print(f"序列数: {len(sequences)}")
    print(f"sensor.yaml 生成数: {total['sensor_yaml_count']}")
    print(f"GT frame 总数: {total['gt_frames']}")
    if args.dry_run:
        print("\n(dry-run — 无实际写入)")


if __name__ == "__main__":
    main()
