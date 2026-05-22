import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


@dataclass
class ImageRecord:
    timestamp_ns: int
    camera_name: str
    image: np.ndarray


@dataclass
class ImuRecord:
    timestamp_ns: int
    gyro: Sequence[float]
    accel: Sequence[float]


@dataclass
class PoseRecord:
    timestamp_ns: int
    position: Sequence[float]
    quaternion_wxyz: Sequence[float]


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_distortion_model(model: str) -> str:
    if model == "radtan":
        return "radial-tangential"
    return model


def write_camera_sensor_yaml(sensor_yaml_path: Path, camera_name: str, camera_config: Dict) -> None:
    camera_from_imu = np.asarray(camera_config["T_cam_imu"], dtype=np.float64).reshape(4, 4)
    body_from_sensor = np.linalg.inv(camera_from_imu)

    sensor = {
        "sensor_type": "camera",
        "comment": f"KAIST-VI {camera_name} exported from ROS bag",
        "T_BS": {
            "cols": 4,
            "rows": 4,
            "data": body_from_sensor.reshape(-1).tolist(),
        },
        "camera_model": camera_config.get("camera_model", "pinhole"),
        "distortion_model": normalize_distortion_model(
            camera_config.get("distortion_model", "")
        ),
        "distortion_coefficients": list(camera_config.get("distortion_coeffs", [])),
        "intrinsics": list(camera_config["intrinsics"]),
        "resolution": list(camera_config["resolution"]),
        "rostopic": camera_config.get("rostopic"),
        "timeshift_cam_imu": camera_config.get("timeshift_cam_imu"),
    }

    sensor_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sensor_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(sensor, f, sort_keys=False)


def write_imu_sensor_yaml(sensor_yaml_path: Path, imu_config: Dict) -> None:
    sensor = {
        "sensor_type": "imu",
        "comment": "KAIST-VI IMU exported from ROS bag",
        "rostopic": imu_config.get("rostopic"),
        "rate_hz": float(imu_config.get("update_rate", 0.0)),
        "gyroscope_noise_density": imu_config.get("gyroscope_noise_density"),
        "gyroscope_random_walk": imu_config.get("gyroscope_random_walk"),
        "accelerometer_noise_density": imu_config.get("accelerometer_noise_density"),
        "accelerometer_random_walk": imu_config.get("accelerometer_random_walk"),
    }

    sensor_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sensor_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(sensor, f, sort_keys=False)


def write_image_csv(csv_path: Path, rows: Iterable[Tuple[int, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["#timestamp [ns]", "filename"])
        for timestamp_ns, filename in rows:
            writer.writerow([int(timestamp_ns), filename])


def write_imu_csv(csv_path: Path, rows: Iterable[Tuple[int, Sequence[float], Sequence[float]]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "#timestamp [ns]",
                "w_RS_S_x [rad s^-1]",
                "w_RS_S_y [rad s^-1]",
                "w_RS_S_z [rad s^-1]",
                "a_RS_S_x [m s^-2]",
                "a_RS_S_y [m s^-2]",
                "a_RS_S_z [m s^-2]",
            ]
        )
        for timestamp_ns, gyro, accel in rows:
            writer.writerow([int(timestamp_ns), *gyro, *accel])


def write_groundtruth_csv(
    csv_path: Path,
    rows: Iterable[Tuple[int, Sequence[float], Sequence[float]]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "#timestamp",
                "p_RS_R_x [m]",
                "p_RS_R_y [m]",
                "p_RS_R_z [m]",
                "q_RS_w []",
                "q_RS_x []",
                "q_RS_y []",
                "q_RS_z []",
                "v_RS_R_x [m s^-1]",
                "v_RS_R_y [m s^-1]",
                "v_RS_R_z [m s^-1]",
                "b_w_RS_S_x [rad s^-1]",
                "b_w_RS_S_y [rad s^-1]",
                "b_w_RS_S_z [rad s^-1]",
                "b_a_RS_S_x [m s^-2]",
                "b_a_RS_S_y [m s^-2]",
                "b_a_RS_S_z [m s^-2]",
            ]
        )
        for timestamp_ns, position, quaternion_wxyz in rows:
            writer.writerow(
                [
                    int(timestamp_ns),
                    *position,
                    *quaternion_wxyz,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )


def ros_time_to_ns(stamp) -> int:
    sec = getattr(stamp, "sec", getattr(stamp, "secs", 0))
    nsec = getattr(stamp, "nanosec", getattr(stamp, "nsec", 0))
    return int(sec) * 1_000_000_000 + int(nsec)


def message_timestamp_ns(message, fallback_timestamp_ns: int) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return int(fallback_timestamp_ns)
    timestamp_ns = ros_time_to_ns(stamp)
    return timestamp_ns if timestamp_ns > 0 else int(fallback_timestamp_ns)


def decode_rosbags_image(message) -> np.ndarray:
    height = int(message.height)
    width = int(message.width)
    encoding = str(message.encoding).lower()
    data = np.frombuffer(message.data, dtype=np.uint8)

    if encoding in ("mono8", "8uc1"):
        return data.reshape(height, int(message.step))[:, :width].copy()
    if encoding in ("bgr8", "rgb8"):
        row = data.reshape(height, int(message.step))[:, : width * 3]
        image = row.reshape(height, width, 3)
        if encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image.copy()
    if encoding in ("mono16", "16uc1"):
        data16 = np.frombuffer(message.data, dtype=np.uint16)
        return data16.reshape(height, int(message.step) // 2)[:, :width].copy()

    raise ValueError(f"Unsupported image encoding: {message.encoding}")


def read_bag_with_rosbags(
    bag_path: Path,
    topic_to_camera: Dict[str, str],
    imu_topic: str,
    gt_topic: Optional[str],
) -> Tuple[List[ImageRecord], List[ImuRecord], List[PoseRecord]]:
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise RuntimeError(
            "Python package 'rosbags' is required to read ROS bags without a ROS install. "
            "Install it with: conda run -n my_vggt_relocation pip install rosbags"
        ) from exc

    image_records: List[ImageRecord] = []
    imu_records: List[ImuRecord] = []
    pose_records: List[PoseRecord] = []
    topics = set(topic_to_camera.keys()) | {imu_topic}
    if gt_topic:
        topics.add(gt_topic)

    with AnyReader([bag_path]) as reader:
        connections = [conn for conn in reader.connections if conn.topic in topics]
        for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
            message = reader.deserialize(rawdata, connection.msgtype)
            msg_timestamp_ns = message_timestamp_ns(message, timestamp_ns)

            if connection.topic in topic_to_camera:
                image_records.append(
                    ImageRecord(
                        timestamp_ns=msg_timestamp_ns,
                        camera_name=topic_to_camera[connection.topic],
                        image=decode_rosbags_image(message),
                    )
                )
            elif connection.topic == imu_topic:
                gyro = message.angular_velocity
                accel = message.linear_acceleration
                imu_records.append(
                    ImuRecord(
                        timestamp_ns=msg_timestamp_ns,
                        gyro=[float(gyro.x), float(gyro.y), float(gyro.z)],
                        accel=[float(accel.x), float(accel.y), float(accel.z)],
                    )
                )
            elif gt_topic and connection.topic == gt_topic:
                position = message.pose.position
                orientation = message.pose.orientation
                pose_records.append(
                    PoseRecord(
                        timestamp_ns=msg_timestamp_ns,
                        position=[
                            float(position.x),
                            float(position.y),
                            float(position.z),
                        ],
                        quaternion_wxyz=[
                            float(orientation.w),
                            float(orientation.x),
                            float(orientation.y),
                            float(orientation.z),
                        ],
                    )
                )

    return image_records, imu_records, pose_records


def read_bag_with_ros1(
    bag_path: Path,
    topic_to_camera: Dict[str, str],
    imu_topic: str,
    gt_topic: Optional[str],
) -> Tuple[List[ImageRecord], List[ImuRecord], List[PoseRecord]]:
    try:
        import rosbag
        from cv_bridge import CvBridge
    except ImportError as exc:
        raise RuntimeError(
            "No ROS bag reader is available. Install 'rosbags' in my_vggt_relocation, "
            "or run inside a ROS1 environment with rosbag and cv_bridge."
        ) from exc

    bridge = CvBridge()
    image_records: List[ImageRecord] = []
    imu_records: List[ImuRecord] = []
    pose_records: List[PoseRecord] = []
    topics = list(topic_to_camera.keys()) + [imu_topic]
    if gt_topic:
        topics.append(gt_topic)

    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, message, timestamp in bag.read_messages(topics=topics):
            timestamp_ns = message_timestamp_ns(message, timestamp.to_nsec())
            if topic in topic_to_camera:
                image_records.append(
                    ImageRecord(
                        timestamp_ns=timestamp_ns,
                        camera_name=topic_to_camera[topic],
                        image=bridge.imgmsg_to_cv2(message, desired_encoding="passthrough"),
                    )
                )
            elif topic == imu_topic:
                gyro = message.angular_velocity
                accel = message.linear_acceleration
                imu_records.append(
                    ImuRecord(
                        timestamp_ns=timestamp_ns,
                        gyro=[float(gyro.x), float(gyro.y), float(gyro.z)],
                        accel=[float(accel.x), float(accel.y), float(accel.z)],
                    )
                )
            elif gt_topic and topic == gt_topic:
                position = message.pose.position
                orientation = message.pose.orientation
                pose_records.append(
                    PoseRecord(
                        timestamp_ns=timestamp_ns,
                        position=[
                            float(position.x),
                            float(position.y),
                            float(position.z),
                        ],
                        quaternion_wxyz=[
                            float(orientation.w),
                            float(orientation.x),
                            float(orientation.y),
                            float(orientation.z),
                        ],
                    )
                )

    return image_records, imu_records, pose_records


def read_bag_records(
    bag_path: Path,
    topic_to_camera: Dict[str, str],
    imu_topic: str,
    gt_topic: Optional[str],
    reader_backend: str,
) -> Tuple[List[ImageRecord], List[ImuRecord], List[PoseRecord]]:
    if reader_backend == "rosbags":
        return read_bag_with_rosbags(bag_path, topic_to_camera, imu_topic, gt_topic)
    if reader_backend == "ros1":
        return read_bag_with_ros1(bag_path, topic_to_camera, imu_topic, gt_topic)

    try:
        return read_bag_with_rosbags(bag_path, topic_to_camera, imu_topic, gt_topic)
    except RuntimeError as rosbags_error:
        try:
            return read_bag_with_ros1(bag_path, topic_to_camera, imu_topic, gt_topic)
        except RuntimeError as ros1_error:
            raise RuntimeError(f"{rosbags_error}\n{ros1_error}") from ros1_error


def bag_sequence_output_dir(output_dir: Path, kaist_vi_dir: Optional[Path], bag_path: Path) -> Path:
    if kaist_vi_dir is not None:
        try:
            rel = bag_path.resolve().relative_to((kaist_vi_dir / "data").resolve())
            return output_dir / rel.parent / bag_path.stem
        except ValueError:
            pass
    return output_dir / bag_path.stem


def discover_bags(kaist_vi_dir: Path) -> List[Path]:
    return sorted((kaist_vi_dir / "data").glob("*/*.bag"))


def write_asl_sequence(
    sequence_dir: Path,
    image_records: Sequence[ImageRecord],
    imu_records: Sequence[ImuRecord],
    pose_records: Sequence[PoseRecord],
    camera_configs: Dict[str, Dict],
    imu_config: Dict,
) -> Dict:
    mav0_dir = sequence_dir / "mav0"
    image_rows: Dict[str, List[Tuple[int, str]]] = {
        camera_name: [] for camera_name in camera_configs
    }

    for camera_name, camera_config in camera_configs.items():
        camera_dir = mav0_dir / camera_name
        (camera_dir / "data").mkdir(parents=True, exist_ok=True)
        write_camera_sensor_yaml(camera_dir / "sensor.yaml", camera_name, camera_config)

    for record in sorted(image_records, key=lambda item: (item.camera_name, item.timestamp_ns)):
        filename = f"{record.timestamp_ns}.png"
        image_path = mav0_dir / record.camera_name / "data" / filename
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), record.image):
            raise RuntimeError(f"Failed to write image: {image_path}")
        image_rows.setdefault(record.camera_name, []).append((record.timestamp_ns, filename))

    for camera_name, rows in image_rows.items():
        write_image_csv(mav0_dir / camera_name / "data.csv", sorted(rows))

    imu0_dir = mav0_dir / "imu0"
    write_imu_csv(
        imu0_dir / "data.csv",
        [
            (record.timestamp_ns, record.gyro, record.accel)
            for record in sorted(imu_records, key=lambda item: item.timestamp_ns)
        ],
    )
    write_imu_sensor_yaml(imu0_dir / "sensor.yaml", imu_config)

    gt_dir = mav0_dir / "state_groundtruth_estimate0"
    write_groundtruth_csv(
        gt_dir / "data.csv",
        [
            (record.timestamp_ns, record.position, record.quaternion_wxyz)
            for record in sorted(pose_records, key=lambda item: item.timestamp_ns)
        ],
    )
    with open(gt_dir / "sensor.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "sensor_type": "ground_truth",
                "comment": "KAIST-VI /pose_transformed exported as EuRoC state_groundtruth_estimate0",
            },
            f,
            sort_keys=False,
        )

    return {
        "camera_counts": {
            camera_name: len(rows) for camera_name, rows in sorted(image_rows.items())
        },
        "imu_count": len(imu_records),
        "groundtruth_count": len(pose_records),
    }


def convert_bag(
    bag_path: Path,
    output_dir: Path,
    camera_configs: Dict[str, Dict],
    imu_config: Dict,
    kaist_vi_dir: Optional[Path],
    reader_backend: str,
    gt_topic: Optional[str],
) -> Dict:
    topic_to_camera = {
        camera_config["rostopic"]: camera_name
        for camera_name, camera_config in camera_configs.items()
    }
    imu_topic = imu_config["rostopic"]
    image_records, imu_records, pose_records = read_bag_records(
        bag_path=bag_path,
        topic_to_camera=topic_to_camera,
        imu_topic=imu_topic,
        gt_topic=gt_topic,
        reader_backend=reader_backend,
    )
    sequence_dir = bag_sequence_output_dir(output_dir, kaist_vi_dir, bag_path)
    stats = write_asl_sequence(
        sequence_dir=sequence_dir,
        image_records=image_records,
        imu_records=imu_records,
        pose_records=pose_records,
        camera_configs=camera_configs,
        imu_config=imu_config,
    )
    return {
        "bag_path": str(bag_path),
        "sequence_dir": str(sequence_dir),
        **stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert KAIST-VI ROS bag files to EuRoC/ASL dataset format."
    )
    parser.add_argument("--kaist_vi_dir", help="Path to local KAIST_VI root.")
    parser.add_argument("--bag_path", help="Path to a single KAIST-VI .bag file.")
    parser.add_argument("--config_dir", help="Path to KAIST-VI config directory.")
    parser.add_argument("--output_dir", required=True, help="Directory to write ASL output.")
    parser.add_argument(
        "--camera_names",
        nargs="+",
        default=["cam0", "cam1"],
        help="Camera names to export from cam-imu.yaml.",
    )
    parser.add_argument(
        "--reader_backend",
        choices=["auto", "rosbags", "ros1"],
        default="auto",
        help="ROS bag reader backend.",
    )
    parser.add_argument(
        "--gt_topic",
        default="/pose_transformed",
        help=(
            "Ground-truth PoseStamped topic to export as "
            "mav0/state_groundtruth_estimate0/data.csv. Use an empty string to skip."
        ),
    )
    args = parser.parse_args()

    if args.kaist_vi_dir is None and args.bag_path is None:
        raise ValueError("Specify either --kaist_vi_dir or --bag_path.")

    kaist_vi_dir = Path(args.kaist_vi_dir).resolve() if args.kaist_vi_dir else None
    if args.config_dir:
        config_dir = Path(args.config_dir).resolve()
    elif kaist_vi_dir is not None:
        config_dir = kaist_vi_dir / "config"
    else:
        raise ValueError("--config_dir is required when using --bag_path without --kaist_vi_dir.")

    cam_imu = load_yaml(config_dir / "cam-imu.yaml")
    imu_config = load_yaml(config_dir / "imu-params.yaml")
    camera_configs = {
        camera_name: cam_imu[camera_name]
        for camera_name in args.camera_names
        if camera_name in cam_imu
    }
    missing_cameras = sorted(set(args.camera_names) - set(camera_configs.keys()))
    if missing_cameras:
        raise ValueError(f"Camera(s) not found in cam-imu.yaml: {missing_cameras}")

    bag_paths = [Path(args.bag_path).resolve()] if args.bag_path else discover_bags(kaist_vi_dir)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_format": "asl",
        "source_dataset": "KAIST_VI",
        "reader_backend": args.reader_backend,
        "camera_names": list(args.camera_names),
        "gt_topic": args.gt_topic or None,
        "bags": [],
    }
    for bag_path in bag_paths:
        print(f"Converting {bag_path}")
        summary["bags"].append(
            convert_bag(
                bag_path=bag_path,
                output_dir=output_dir,
                camera_configs=camera_configs,
                imu_config=imu_config,
                kaist_vi_dir=kaist_vi_dir,
                reader_backend=args.reader_backend,
                gt_topic=args.gt_topic or None,
            )
        )

    with open(output_dir / "conversion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Converted bags: {len(summary['bags'])}")
    print(f"ASL output directory: {output_dir}")


if __name__ == "__main__":
    main()
