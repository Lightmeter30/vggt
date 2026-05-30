import gzip
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import yaml

from training.data.preprocess.undistort_uma_vi_bumblebee import process_uma_vi


def _write_image(path: Path, value: int, size: tuple[int, int] = (24, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = size
    image = np.full((height, width, 3), value, dtype=np.uint8)
    image[:, : width // 2, 0] = value // 2
    assert cv2.imwrite(str(path), image)


def _write_sensor(
    path: Path,
    distortion_model: str,
    intrinsics: list[float] | None = None,
    resolution: list[int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sensor = {
        "sensor_type": "camera",
        "comment": "synthetic UMA camera",
        "T_BS": {"cols": 4, "rows": 4, "data": np.eye(4).reshape(-1).tolist()},
        "rate_hz": 10.0,
        "resolution": resolution or [32, 24],
        "camera_model": "pinhole",
        "intrinsics": intrinsics or [20.0, 20.5, 16.0, 12.0],
        "distortion_model": distortion_model,
        "distortion_coefficients": [0.02, -0.005, 0.001, 0.0],
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(sensor, f, sort_keys=False)


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _intrinsics_matrix_from_sensor(sensor: dict) -> list[list[float]]:
    fx, fy, cx, cy = sensor["intrinsics"]
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


class TestUndistortUmaViBumblebee(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.uma_dir = Path(self.temp_dir.name) / "UMA_VI"
        self.sequence_dir = self.uma_dir / "class-eng_2019-02-07-14-14-09_Indoor"
        self.anno_dir = self.uma_dir / "anno"

        for camera_name, model in (
            ("cam0", "equidistant"),
            ("cam1", "equidistant"),
            ("cam2", "radial-tangential"),
        ):
            raw_camera_dir = self.sequence_dir / camera_name
            mav_camera_dir = self.sequence_dir / "mav0" / camera_name
            _write_image(raw_camera_dir / "data" / "1.png", 80)
            _write_sensor(mav_camera_dir / "sensor.yaml", model)
            mav_camera_dir.mkdir(parents=True, exist_ok=True)
            (mav_camera_dir / "data").symlink_to(Path("..") / ".." / camera_name / "data")

        sensor = {
            "intrinsics": [[20.0, 0.0, 16.0], [0.0, 20.5, 12.0], [0.0, 0.0, 1.0]],
            "distortion": [0.02, -0.005, 0.001, 0.0],
            "undistorted_intrinsics": [[20.0, 0.0, 16.0], [0.0, 20.5, 12.0], [0.0, 0.0, 1.0]],
            "image_size": [32, 24],
            "distortion_model": "equidistant",
        }
        payload = {
            "uma_vi/class-eng_2019-02-07-14-14-09_Indoor/cam0": {
                "sequence_name": "class-eng_2019-02-07-14-14-09_Indoor",
                "sequence_path": "class-eng_2019-02-07-14-14-09_Indoor",
                "camera_name": "cam0",
                "sensor": sensor,
                "frames": [],
                "imu_data": None,
            }
        }
        self.anno_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.anno_dir / "class-eng_2019-02-07-14-14-09_Indoor.jgz", "wt", encoding="utf-8") as f:
            json.dump(payload, f)
        manifest = {
            "schema_version": "vi_pose_v1",
            "dataset": "uma_vi",
            "split_policy": "configured_in_training",
            "camera_names": ["cam0", "cam1", "cam2"],
            "max_pose_time_diff_ns": 10_000_000,
            "sequences": {
                "class-eng_2019-02-07-14-14-09_Indoor": {
                    "file": "class-eng_2019-02-07-14-14-09_Indoor.jgz",
                    "sequence_path": "class-eng_2019-02-07-14-14-09_Indoor",
                    "frame_count": 1,
                    "camera_names": ["cam0", "cam1", "cam2"],
                    "distortion_models": {
                        "cam0": "equidistant",
                        "cam1": "equidistant",
                        "cam2": "radial-tangential",
                    },
                }
            },
        }
        (self.anno_dir / "sequence_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (self.anno_dir / "summary.json").write_text(
            json.dumps({"dataset_name": "uma_vi", "sequence_manifest": manifest}),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dry_run_does_not_create_origin_or_modify_sensor(self):
        before = _read_yaml(self.sequence_dir / "mav0" / "cam0" / "sensor.yaml")

        result = process_uma_vi(self.uma_dir, dry_run=True)

        after = _read_yaml(self.sequence_dir / "mav0" / "cam0" / "sensor.yaml")
        self.assertEqual(result["processed_images"], 0)
        self.assertFalse((self.uma_dir / "origin").exists())
        self.assertEqual(after, before)

    def test_process_backs_up_images_and_updates_bumblebee_metadata(self):
        original = cv2.imread(str(self.sequence_dir / "cam0" / "data" / "1.png"), cv2.IMREAD_COLOR)

        result = process_uma_vi(self.uma_dir)

        backup_path = (
            self.uma_dir
            / "origin"
            / self.sequence_dir.name
            / "cam0"
            / "data"
            / "1.png"
        )
        self.assertTrue(backup_path.is_file())
        self.assertTrue(np.array_equal(cv2.imread(str(backup_path), cv2.IMREAD_COLOR), original))
        self.assertEqual(result["processed_cameras"], 2)
        self.assertEqual(result["skipped_cameras"], 0)
        self.assertEqual(result["processed_images"], 2)

        cam0_sensor = _read_yaml(self.sequence_dir / "mav0" / "cam0" / "sensor.yaml")
        cam2_sensor = _read_yaml(self.sequence_dir / "mav0" / "cam2" / "sensor.yaml")
        self.assertEqual(cam0_sensor["distortion_model"], "radial-tangential")
        self.assertEqual(cam0_sensor["distortion_coefficients"], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(cam2_sensor["distortion_model"], "radial-tangential")
        self.assertEqual(cam2_sensor["distortion_coefficients"], [0.02, -0.005, 0.001, 0.0])

        with gzip.open(self.anno_dir / "class-eng_2019-02-07-14-14-09_Indoor.jgz", "rt", encoding="utf-8") as f:
            annotation = json.load(f)
        sensor = annotation["uma_vi/class-eng_2019-02-07-14-14-09_Indoor/cam0"]["sensor"]
        self.assertEqual(sensor["distortion_model"], "radial-tangential")
        self.assertEqual(sensor["distortion"], [0.0, 0.0, 0.0, 0.0])

        manifest = json.loads((self.anno_dir / "sequence_manifest.json").read_text(encoding="utf-8"))
        distortion_models = manifest["sequences"][self.sequence_dir.name]["distortion_models"]
        self.assertEqual(distortion_models["cam0"], "radial-tangential")
        self.assertEqual(distortion_models["cam1"], "radial-tangential")
        self.assertEqual(distortion_models["cam2"], "radial-tangential")

        summary = json.loads((self.anno_dir / "summary.json").read_text(encoding="utf-8"))
        summary_models = summary["sequence_manifest"]["sequences"][self.sequence_dir.name]["distortion_models"]
        self.assertEqual(summary_models["cam0"], "radial-tangential")
        self.assertEqual(summary_models["cam1"], "radial-tangential")

    def test_process_is_idempotent_and_keeps_first_origin_backup(self):
        process_uma_vi(self.uma_dir)
        backup_path = self.uma_dir / "origin" / self.sequence_dir.name / "cam0" / "data" / "1.png"
        first_backup = cv2.imread(str(backup_path), cv2.IMREAD_COLOR)

        _write_image(self.sequence_dir / "cam0" / "data" / "1.png", 10)
        process_uma_vi(self.uma_dir)

        second_backup = cv2.imread(str(backup_path), cv2.IMREAD_COLOR)
        self.assertTrue(np.array_equal(second_backup, first_backup))

    def test_process_repairs_unreadable_origin_backup_from_current_image(self):
        backup_path = self.uma_dir / "origin" / self.sequence_dir.name / "cam0" / "data" / "1.png"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_bytes(b"partial png")

        process_uma_vi(self.uma_dir)

        repaired = cv2.imread(str(backup_path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(repaired)

    def test_updates_annotation_intrinsics_per_sequence_camera(self):
        second_sequence = self.uma_dir / "warehouse_02"
        raw_camera_dir = second_sequence / "cam0"
        mav_camera_dir = second_sequence / "mav0" / "cam0"
        _write_image(raw_camera_dir / "data" / "1.png", 120, size=(30, 40))
        _write_sensor(
            mav_camera_dir / "sensor.yaml",
            "equidistant",
            intrinsics=[35.0, 36.0, 20.0, 15.0],
            resolution=[40, 30],
        )
        mav_camera_dir.mkdir(parents=True, exist_ok=True)
        (mav_camera_dir / "data").symlink_to(Path("..") / ".." / "cam0" / "data")

        second_sensor = {
            "intrinsics": [[35.0, 0.0, 20.0], [0.0, 36.0, 15.0], [0.0, 0.0, 1.0]],
            "distortion": [0.02, -0.005, 0.001, 0.0],
            "undistorted_intrinsics": [[35.0, 0.0, 20.0], [0.0, 36.0, 15.0], [0.0, 0.0, 1.0]],
            "image_size": [40, 30],
            "distortion_model": "equidistant",
        }
        second_payload = {
            "uma_vi/warehouse_02/cam0": {
                "sequence_name": "warehouse_02",
                "sequence_path": "warehouse_02",
                "camera_name": "cam0",
                "sensor": second_sensor,
                "frames": [],
                "imu_data": None,
            }
        }
        with gzip.open(self.anno_dir / "warehouse_02.jgz", "wt", encoding="utf-8") as f:
            json.dump(second_payload, f)

        process_uma_vi(self.uma_dir, cameras=("cam0",))

        first_sensor_yaml = _read_yaml(self.sequence_dir / "mav0" / "cam0" / "sensor.yaml")
        second_sensor_yaml = _read_yaml(second_sequence / "mav0" / "cam0" / "sensor.yaml")
        with gzip.open(self.anno_dir / f"{self.sequence_dir.name}.jgz", "rt", encoding="utf-8") as f:
            first_annotation = json.load(f)
        with gzip.open(self.anno_dir / "warehouse_02.jgz", "rt", encoding="utf-8") as f:
            second_annotation = json.load(f)

        first_annotation_sensor = first_annotation[f"uma_vi/{self.sequence_dir.name}/cam0"]["sensor"]
        second_annotation_sensor = second_annotation["uma_vi/warehouse_02/cam0"]["sensor"]
        self.assertEqual(
            first_annotation_sensor["intrinsics"],
            _intrinsics_matrix_from_sensor(first_sensor_yaml),
        )
        self.assertEqual(
            second_annotation_sensor["intrinsics"],
            _intrinsics_matrix_from_sensor(second_sensor_yaml),
        )
        self.assertNotEqual(
            first_annotation_sensor["intrinsics"],
            second_annotation_sensor["intrinsics"],
        )


if __name__ == "__main__":
    unittest.main()
