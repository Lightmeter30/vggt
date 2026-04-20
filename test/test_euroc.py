from pathlib import Path
from types import SimpleNamespace
import csv
import sys
import tempfile
import unittest

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from data.datasets.euroc import EurocDataset


def _make_common_conf():
    return SimpleNamespace(
        img_size=32,
        patch_size=8,
        augs=SimpleNamespace(scales=None),
        rescale=True,
        rescale_aug=False,
        landscape_check=False,
        debug=False,
        training=False,
        get_nearby=False,
        inside_random=False,
        allow_duplicate_img=False,
    )


def _write_csv(csv_path: Path, header, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_sensor_yaml(sensor_yaml_path: Path, intrinsics, t_bs):
    sensor_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    sensor_dict = {
        "sensor_type": "camera",
        "comment": "synthetic camera",
        "T_BS": {
            "cols": 4,
            "rows": 4,
            "data": np.asarray(t_bs, dtype=float).reshape(-1).tolist(),
        },
        "rate_hz": 20,
        "resolution": [64, 64],
        "camera_model": "pinhole",
        "intrinsics": list(intrinsics),
        "distortion_model": "radial-tangential",
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
    }
    with open(sensor_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(sensor_dict, f, sort_keys=False)


def _write_sequence(root: Path, relative_sequence: str, frame_count: int = 3):
    sequence_root = root / relative_sequence / "mav0"
    cam0_dir = sequence_root / "cam0"
    cam1_dir = sequence_root / "cam1"
    imu0_dir = sequence_root / "imu0"
    gt_dir = sequence_root / "state_groundtruth_estimate0"

    timestamps = [1_000_000_000 + idx * 50_000_000 for idx in range(frame_count)]

    t_bs_cam0 = np.eye(4, dtype=np.float32)
    t_bs_cam1 = np.eye(4, dtype=np.float32)
    t_bs_cam1[0, 3] = 0.1

    _write_sensor_yaml(cam0_dir / "sensor.yaml", [40.0, 40.0, 32.0, 32.0], t_bs_cam0)
    _write_sensor_yaml(cam1_dir / "sensor.yaml", [41.0, 41.0, 31.5, 31.5], t_bs_cam1)

    cam_rows = []
    for idx, timestamp in enumerate(timestamps):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[..., 0] = 10 * idx
        image[..., 1] = 20 + idx
        image[..., 2] = 50
        file_name = f"{timestamp}.png"
        cam0_path = cam0_dir / "data" / file_name
        cam1_path = cam1_dir / "data" / file_name
        cam0_path.parent.mkdir(parents=True, exist_ok=True)
        cam1_path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(cam0_path), image)
        assert cv2.imwrite(str(cam1_path), image)
        cam_rows.append([timestamp, file_name])

    _write_csv(cam0_dir / "data.csv", ["#timestamp [ns]", "filename"], cam_rows)
    _write_csv(cam1_dir / "data.csv", ["#timestamp [ns]", "filename"], cam_rows)

    gt_rows = []
    for idx, timestamp in enumerate(timestamps):
        gt_rows.append(
            [
                timestamp,
                float(idx),
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
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
    _write_csv(
        gt_dir / "data.csv",
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
        ],
        gt_rows,
    )

    imu_rows = []
    for timestamp in range(timestamps[0] - 25_000_000, timestamps[-1] + 25_000_001, 10_000_000):
        imu_rows.append([timestamp, 0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
    _write_csv(
        imu0_dir / "data.csv",
        [
            "#timestamp [ns]",
            "w_RS_S_x [rad s^-1]",
            "w_RS_S_y [rad s^-1]",
            "w_RS_S_z [rad s^-1]",
            "a_RS_S_x [m s^-2]",
            "a_RS_S_y [m s^-2]",
            "a_RS_S_z [m s^-2]",
        ],
        imu_rows,
    )

    return root / relative_sequence


class TestEurocDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.synthetic_euroc_root = Path(self.temp_dir.name)
        _write_sequence(self.synthetic_euroc_root, "machine_hall/MH_01_easy")
        _write_sequence(self.synthetic_euroc_root, "machine_hall/MH_02_easy")
        _write_sequence(self.synthetic_euroc_root, "vicon_room1/V1_01_easy")

        invalid_root = self.synthetic_euroc_root / "invalid_seq" / "mav0" / "cam0"
        invalid_root.mkdir(parents=True, exist_ok=True)
        _write_csv(invalid_root / "data.csv", ["#timestamp [ns]", "filename"], [[1, "1.png"]])

        self.common_conf = _make_common_conf()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_discover_sequences_and_split(self):
        train_dataset = EurocDataset(
            common_conf=self.common_conf,
            split="train",
            EUROC_DIR=str(self.synthetic_euroc_root),
            min_num_images=2,
            train_split_ratio=2 / 3,
            undistort_images=False,
        )
        test_dataset = EurocDataset(
            common_conf=self.common_conf,
            split="test",
            EUROC_DIR=str(self.synthetic_euroc_root),
            min_num_images=2,
            train_split_ratio=2 / 3,
            undistort_images=False,
        )

        self.assertEqual(
            train_dataset.sequence_list,
            ["machine_hall/MH_01_easy", "machine_hall/MH_02_easy"],
        )
        self.assertEqual(test_dataset.sequence_list, ["vicon_room1/V1_01_easy"])
        self.assertEqual(train_dataset.total_frame_num, 6)
        self.assertEqual(test_dataset.total_frame_num, 3)

    def test_helper_loaders_and_timestamp_matching(self):
        dataset = EurocDataset(
            common_conf=self.common_conf,
            split="train",
            EUROC_DIR=str(self.synthetic_euroc_root),
            min_num_images=2,
            train_split_ratio=1.0,
            undistort_images=False,
        )

        seq_root = self.synthetic_euroc_root / "machine_hall" / "MH_01_easy" / "mav0"
        intrinsics, distortion, body_from_camera = dataset._load_camera_sensor(seq_root / "cam0" / "sensor.yaml")
        gt_timestamps, world_from_body = dataset._load_groundtruth(seq_root / "state_groundtruth_estimate0" / "data.csv")
        imu_data = dataset._load_imu_data(seq_root / "imu0" / "data.csv")
        nearest_idx, nearest_dt = dataset._find_nearest_timestamp(gt_timestamps, gt_timestamps[1] + 1)

        self.assertEqual(intrinsics.shape, (3, 3))
        self.assertTrue(np.allclose(intrinsics[0, 0], 40.0))
        self.assertEqual(distortion.shape, (4,))
        self.assertTrue(np.allclose(body_from_camera, np.eye(4)))

        self.assertEqual(gt_timestamps.shape, (3,))
        self.assertEqual(world_from_body.shape, (3, 4, 4))
        self.assertTrue(np.allclose(world_from_body[0], np.eye(4)))
        self.assertTrue(np.isclose(world_from_body[1, 0, 3], 1.0))

        self.assertEqual(imu_data["timestamps_ns"].ndim, 1)
        self.assertEqual(imu_data["gyro"].shape[1], 3)
        self.assertEqual(imu_data["accel"].shape[1], 3)

        self.assertEqual(nearest_idx, 1)
        self.assertEqual(nearest_dt, 1)

    def test_multi_camera_sequences_and_frame_entries(self):
        dataset = EurocDataset(
            common_conf=self.common_conf,
            split="train",
            EUROC_DIR=str(self.synthetic_euroc_root),
            min_num_images=2,
            train_split_ratio=1.0,
            camera_names=("cam0", "cam1"),
            undistort_images=False,
        )

        self.assertIn("machine_hall/MH_01_easy:cam0", dataset.sequence_list)
        self.assertIn("machine_hall/MH_01_easy:cam1", dataset.sequence_list)

        cam0_frames = dataset.data_store["machine_hall/MH_01_easy:cam0"]["frames"]
        cam1_frames = dataset.data_store["machine_hall/MH_01_easy:cam1"]["frames"]

        self.assertEqual(len(cam0_frames), 3)
        self.assertEqual(len(cam1_frames), 3)
        self.assertEqual(cam0_frames[0]["extrinsics"].shape, (3, 4))
        self.assertTrue(
            np.allclose(
                cam0_frames[0]["extrinsics"],
                np.hstack([np.eye(3), np.zeros((3, 1))]),
                atol=1e-6,
            )
        )
        self.assertTrue(np.isclose(cam1_frames[0]["extrinsics"][0, 3], -0.1, atol=1e-6))

    def test_get_data_returns_expected_batch_and_imu_windows(self):
        dataset = EurocDataset(
            common_conf=self.common_conf,
            split="train",
            EUROC_DIR=str(self.synthetic_euroc_root),
            min_num_images=2,
            train_split_ratio=1.0,
            load_imu=True,
            undistort_images=False,
        )

        batch = dataset.get_data(
            seq_name="machine_hall/MH_01_easy",
            ids=np.array([0, 1]),
            img_per_seq=2,
            aspect_ratio=1.0,
        )

        self.assertEqual(batch["seq_name"], "euroc_machine_hall/MH_01_easy")
        self.assertEqual(batch["frame_num"], 2)
        self.assertEqual(batch["ids"].tolist(), [0, 1])

        expected_keys = {
            "seq_name",
            "ids",
            "frame_num",
            "images",
            "depths",
            "extrinsics",
            "intrinsics",
            "cam_points",
            "world_points",
            "point_masks",
            "original_sizes",
            "imu_windows",
        }
        self.assertEqual(set(batch.keys()), expected_keys)

        self.assertEqual(len(batch["images"]), 2)
        self.assertEqual(batch["images"][0].shape, (32, 32, 3))
        self.assertEqual(batch["depths"][0].shape, (32, 32))
        self.assertEqual(batch["extrinsics"][0].shape, (3, 4))
        self.assertEqual(batch["intrinsics"][0].shape, (3, 3))
        self.assertEqual(batch["cam_points"][0].shape, (32, 32, 3))
        self.assertEqual(batch["world_points"][0].shape, (32, 32, 3))
        self.assertEqual(batch["point_masks"][0].shape, (32, 32))

        self.assertEqual(int(np.count_nonzero(batch["depths"][0])), 1)
        self.assertEqual(int(batch["point_masks"][0].sum()), 1)

        self.assertEqual(len(batch["imu_windows"]), 2)
        self.assertGreater(len(batch["imu_windows"][0]["timestamps_ns"]), 0)
        self.assertEqual(batch["imu_windows"][0]["gyro"].shape[1], 3)
        self.assertEqual(batch["imu_windows"][0]["accel"].shape[1], 3)


if __name__ == "__main__":
    unittest.main()
