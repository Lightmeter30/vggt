from pathlib import Path
from types import SimpleNamespace
import csv
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

import cv2
import numpy as np
from torch.utils.data._utils.collate import default_collate
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from data.datasets.asl import ASLDataset
from data.composed_dataset import ComposedDataset
from training.data.preprocess import generate_euroc_annotations as gen_euroc


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


def _make_composed_common_conf():
    return SimpleNamespace(
        img_size=32,
        patch_size=8,
        fix_img_num=-1,
        fix_aspect_ratio=1.0,
        load_track=False,
        track_num=16,
        training=False,
        inside_random=False,
        rescale=True,
        rescale_aug=False,
        landscape_check=False,
        debug=False,
        get_nearby=False,
        allow_duplicate_img=False,
        augs=SimpleNamespace(
            scales=None,
            cojitter=False,
            cojitter_ratio=0.0,
            color_jitter=None,
            gray_scale=False,
            gau_blur=False,
        ),
    )


def _write_csv(csv_path: Path, header, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_sensor_yaml(
    sensor_yaml_path: Path,
    intrinsics,
    t_bs,
    distortion_model: str = "radial-tangential",
    distortion_coefficients=None,
):
    sensor_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    if distortion_coefficients is None:
        distortion_coefficients = [0.0, 0.0, 0.0, 0.0]
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
        "distortion_model": distortion_model,
        "distortion_coefficients": list(distortion_coefficients),
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


class TestASLDataset(unittest.TestCase):
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
        self.annotation_dir = self.synthetic_euroc_root / "anno"
        self.annotation_dir.mkdir(parents=True, exist_ok=True)
        annotation, _ = gen_euroc.build_asl_annotations(
            asl_dir=self.synthetic_euroc_root,
            dataset_name="euroc",
            sequence_dirs=gen_euroc.discover_sequences(self.synthetic_euroc_root),
            camera_names=("cam0", "cam1"),
            max_pose_time_diff_ns=10_000_000,
        )
        gen_euroc.write_sequence_outputs(
            output_dir=self.annotation_dir,
            annotation=annotation,
            camera_names=("cam0", "cam1"),
            max_pose_time_diff_ns=10_000_000,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_discover_sequences_and_split(self):
        train_dataset = ASLDataset(
            common_conf=self.common_conf,
            split="train",
            ASL_DIR=str(self.synthetic_euroc_root),
            ASL_ANNOTATION_DIR=str(self.annotation_dir),
            min_num_images=2,
            camera_names=("cam0",),
            sequence_names=("MH_01_easy", "MH_02_easy"),
            undistort_images=False,
        )
        val_dataset = ASLDataset(
            common_conf=self.common_conf,
            split="val",
            ASL_DIR=str(self.synthetic_euroc_root),
            ASL_ANNOTATION_DIR=str(self.annotation_dir),
            min_num_images=2,
            camera_names=("cam0",),
            sequence_names=("V1_01_easy",),
            undistort_images=False,
        )

        self.assertEqual(
            train_dataset.sequence_list,
            ["euroc/MH_01_easy/cam0", "euroc/MH_02_easy/cam0"],
        )
        self.assertEqual(val_dataset.sequence_list, ["euroc/V1_01_easy/cam0"])
        self.assertEqual(train_dataset.total_frame_num, 6)
        self.assertEqual(val_dataset.total_frame_num, 3)

    def test_eval_length_uses_filtered_sequence_count(self):
        val_dataset = ASLDataset(
            common_conf=self.common_conf,
            split="val",
            ASL_DIR=str(self.synthetic_euroc_root),
            ASL_ANNOTATION_DIR=str(self.annotation_dir),
            min_num_images=2,
            len_test=10000,
            camera_names=("cam0",),
            sequence_names=("V1_01_easy",),
            undistort_images=False,
        )

        self.assertEqual(len(val_dataset), 1)
        self.assertEqual(len(val_dataset), val_dataset.sequence_list_len)

    def test_deserializes_vi_schema_sensor_frames_and_imu(self):
        dataset = ASLDataset(
            common_conf=self.common_conf,
            split="train",
            ASL_DIR=str(self.synthetic_euroc_root),
            ASL_ANNOTATION_DIR=str(self.annotation_dir),
            min_num_images=2,
            camera_names=("cam0",),
            sequence_names=("MH_01_easy", "MH_02_easy"),
            undistort_images=False,
        )

        sequence = dataset.data_store["euroc/MH_01_easy/cam0"]
        sensor = sequence["sensor"]
        frames = sequence["frames"]
        imu_data = sequence["imu_data"]

        self.assertEqual(sequence["dataset"], "euroc")
        self.assertEqual(sequence["sequence_name"], "MH_01_easy")
        self.assertEqual(sequence["split"], "train")

        self.assertEqual(sensor["intrinsics"].shape, (3, 3))
        self.assertTrue(np.allclose(sensor["intrinsics"][0, 0], 40.0))
        self.assertEqual(sensor["distortion"].shape, (4,))
        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[1]["extrinsics"].shape, (3, 4))
        self.assertTrue(np.isclose(frames[1]["extrinsics"][0, 3], -1.0))

        self.assertEqual(imu_data["timestamps_ns"].ndim, 1)
        self.assertEqual(imu_data["gyro"].shape[1], 3)
        self.assertEqual(imu_data["accel"].shape[1], 3)

    def test_multi_camera_sequences_and_frame_entries(self):
        dataset = ASLDataset(
            common_conf=self.common_conf,
            split="train",
            ASL_DIR=str(self.synthetic_euroc_root),
            ASL_ANNOTATION_DIR=str(self.annotation_dir),
            min_num_images=2,
            camera_names=("cam0", "cam1"),
            sequence_names=("MH_01_easy", "MH_02_easy"),
            undistort_images=False,
        )

        self.assertIn("euroc/MH_01_easy/cam0", dataset.sequence_list)
        self.assertIn("euroc/MH_01_easy/cam1", dataset.sequence_list)

        cam0_frames = dataset.data_store["euroc/MH_01_easy/cam0"]["frames"]
        cam1_frames = dataset.data_store["euroc/MH_01_easy/cam1"]["frames"]

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
        dataset = ASLDataset(
            common_conf=self.common_conf,
            split="train",
            ASL_DIR=str(self.synthetic_euroc_root),
            ASL_ANNOTATION_DIR=str(self.annotation_dir),
            min_num_images=2,
            camera_names=("cam0",),
            sequence_names=("MH_01_easy", "MH_02_easy"),
            load_imu=True,
            imu_window_ns=20_000_000,
            imu_num_samples=5,
            undistort_images=False,
        )

        batch = dataset.get_data(
            seq_name="euroc/MH_01_easy/cam0",
            ids=np.array([0, 1]),
            img_per_seq=2,
            aspect_ratio=1.0,
        )

        self.assertEqual(batch["seq_name"], "euroc_euroc/MH_01_easy/cam0")
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
            "imu_window_masks",
            "timestamps_ns",
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

        self.assertEqual(batch["imu_windows"].shape, (2, 5, 6))
        self.assertEqual(batch["imu_window_masks"].shape, (2, 5))
        self.assertEqual(batch["timestamps_ns"].tolist(), [1_000_000_000, 1_050_000_000])
        self.assertTrue(batch["imu_window_masks"][0, 1:4].all())
        self.assertFalse(batch["imu_window_masks"][0, 0])
        self.assertFalse(batch["imu_window_masks"][0, 4])
        self.assertTrue(
            np.allclose(batch["imu_windows"][0, 2], [0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
        )

    def test_get_nearby_returns_time_ordered_imu_sequence(self):
        common_conf = _make_common_conf()
        common_conf.get_nearby = True
        dataset = ASLDataset(
            common_conf=common_conf,
            split="train",
            ASL_DIR=str(self.synthetic_euroc_root),
            ASL_ANNOTATION_DIR=str(self.annotation_dir),
            min_num_images=2,
            camera_names=("cam0",),
            sequence_names=("MH_01_easy", "MH_02_easy"),
            load_imu=True,
            imu_window_ns=20_000_000,
            imu_num_samples=5,
            undistort_images=False,
        )

        with patch("data.base_dataset.np.random.choice", return_value=np.array([2, 0])):
            batch = dataset.get_data(
                seq_name="euroc/MH_01_easy/cam0",
                ids=np.array([1, 1, 1]),
                img_per_seq=3,
                aspect_ratio=1.0,
            )

        self.assertEqual(batch["ids"].tolist(), [0, 1, 2])
        self.assertEqual(
            batch["timestamps_ns"].tolist(),
            [1_000_000_000, 1_050_000_000, 1_100_000_000],
        )

    def test_composed_dataset_collates_fixed_imu_tensors(self):
        common_conf = _make_composed_common_conf()
        composed = ComposedDataset(
            dataset_configs=[
                {
                    "_target_": "data.datasets.asl.ASLDataset",
                    "split": "train",
                    "ASL_DIR": str(self.synthetic_euroc_root),
                    "ASL_ANNOTATION_DIR": str(self.annotation_dir),
                    "min_num_images": 2,
                    "camera_names": ("cam0",),
                    "sequence_names": ("MH_01_easy", "MH_02_easy"),
                    "load_imu": True,
                    "imu_window_ns": 20_000_000,
                    "imu_num_samples": 5,
                    "undistort_images": False,
                }
            ],
            common_config=common_conf,
        )

        sample = composed[(0, 2, 1.0)]
        self.assertEqual(sample["imu_windows"].shape, (2, 5, 6))
        self.assertEqual(sample["imu_window_masks"].shape, (2, 5))
        self.assertEqual(sample["timestamps_ns"].shape, (2,))

        collated = default_collate([sample, sample])
        self.assertEqual(collated["imu_windows"].shape, (2, 2, 5, 6))
        self.assertEqual(collated["imu_window_masks"].shape, (2, 2, 5))
        self.assertEqual(collated["timestamps_ns"].shape, (2, 2))

    def test_equidistant_camera_uses_fisheye_undistortion(self):
        sensor_path = (
            self.synthetic_euroc_root
            / "machine_hall"
            / "MH_01_easy"
            / "mav0"
            / "cam0"
            / "sensor.yaml"
        )
        _write_sensor_yaml(
            sensor_path,
            [40.0, 40.0, 32.0, 32.0],
            np.eye(4, dtype=np.float32),
            distortion_model="equidistant",
            distortion_coefficients=[0.01, -0.001, 0.0001, 0.0],
        )
        annotation, _ = gen_euroc.build_asl_annotations(
            asl_dir=self.synthetic_euroc_root,
            dataset_name="uma_vi",
            sequence_dirs=[
                self.synthetic_euroc_root / "machine_hall" / "MH_01_easy",
            ],
            camera_names=("cam0",),
            max_pose_time_diff_ns=10_000_000,
        )
        gen_euroc.write_sequence_outputs(
            output_dir=self.annotation_dir,
            annotation=annotation,
            camera_names=("cam0",),
            max_pose_time_diff_ns=10_000_000,
        )
        dataset = ASLDataset(
            common_conf=self.common_conf,
            split="train",
            dataset_name="uma_vi",
            ASL_DIR=str(self.synthetic_euroc_root),
            ASL_ANNOTATION_DIR=str(self.annotation_dir),
            min_num_images=2,
            camera_names=("cam0",),
            sequence_names=("MH_01_easy",),
            undistort_images=True,
        )

        with patch("cv2.fisheye.undistortImage", Mock(side_effect=lambda image, *args, **kwargs: image)) as undistort:
            batch = dataset.get_data(
                seq_name="uma_vi/MH_01_easy/cam0",
                ids=np.array([0, 1]),
                img_per_seq=2,
                aspect_ratio=1.0,
            )

        self.assertEqual(batch["frame_num"], 2)
        self.assertTrue(undistort.called)


if __name__ == "__main__":
    unittest.main()
