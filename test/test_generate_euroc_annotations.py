import csv
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.data.preprocess import generate_euroc_annotations as gen_euroc
from training.data.preprocess import inspect_vi_dataset


def _write_csv(csv_path: Path, header, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_sensor_yaml(sensor_yaml_path: Path, t_bs):
    sensor_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    sensor_dict = {
        "sensor_type": "camera",
        "comment": "synthetic ASL camera",
        "T_BS": {
            "cols": 4,
            "rows": 4,
            "data": np.asarray(t_bs, dtype=float).reshape(-1).tolist(),
        },
        "rate_hz": 20,
        "resolution": [16, 16],
        "camera_model": "pinhole",
        "intrinsics": [12.0, 12.0, 8.0, 8.0],
        "distortion_model": "radial-tangential",
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
    }
    with open(sensor_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(sensor_dict, f, sort_keys=False)


def _write_asl_sequence(root: Path, relative_sequence: str):
    sequence_root = root / relative_sequence / "mav0"
    cam0_dir = sequence_root / "cam0"
    gt_dir = sequence_root / "state_groundtruth_estimate0"
    imu0_dir = sequence_root / "imu0"

    timestamps = [1_000_000_000, 1_050_000_000]
    _write_sensor_yaml(cam0_dir / "sensor.yaml", np.eye(4, dtype=np.float32))

    cam_rows = []
    for idx, timestamp in enumerate(timestamps):
        image = np.full((16, 16, 3), 30 + idx, dtype=np.uint8)
        file_name = f"{timestamp}.png"
        image_path = cam0_dir / "data" / file_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(image_path), image)
        cam_rows.append([timestamp, file_name])
    _write_csv(cam0_dir / "data.csv", ["#timestamp [ns]", "filename"], cam_rows)

    gt_rows = [
        [timestamps[0], 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [timestamps[1], 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ]
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
        ],
        gt_rows,
    )
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
        [[timestamps[0], 0.1, 0.2, 0.3, 1.0, 2.0, 3.0]],
    )


class TestGenerateEurocAnnotations(unittest.TestCase):
    def test_main_writes_per_sequence_jgz_with_asl_manifest_and_w2c_extrinsics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            euroc_dir = root / "euroc"
            output_dir = root / "anno"
            _write_asl_sequence(euroc_dir, "machine_hall/MH_01_easy")
            _write_asl_sequence(euroc_dir, "machine_hall/MH_05_difficult")
            _write_asl_sequence(euroc_dir, "vicon_room1/V1_01_easy")
            _write_asl_sequence(euroc_dir, "vicon_room2/V2_02_medium")

            argv = [
                "generate_euroc_annotations.py",
                "--asl_dir",
                str(euroc_dir),
                "--output_dir",
                str(output_dir),
                "--dataset_name",
                "euroc",
            ]
            with mock.patch.object(sys, "argv", argv):
                gen_euroc.main()

            jgz_names = sorted(path.name for path in output_dir.glob("*.jgz"))
            self.assertEqual(
                jgz_names,
                [
                    "machine_hall__MH_01_easy.jgz",
                    "machine_hall__MH_05_difficult.jgz",
                    "vicon_room1__V1_01_easy.jgz",
                    "vicon_room2__V2_02_medium.jgz",
                ],
            )
            self.assertTrue((output_dir / "sequence_manifest.json").is_file())
            self.assertTrue((output_dir / "schema_version.json").is_file())

            with gzip.open(output_dir / "machine_hall__MH_01_easy.jgz", "rt", encoding="utf-8") as f:
                sequence_payload = json.load(f)

            self.assertEqual(list(sequence_payload.keys()), ["euroc/MH_01_easy/cam0"])
            gen_euroc.validate_vi_annotation(sequence_payload)

            sequence = sequence_payload["euroc/MH_01_easy/cam0"]
            self.assertEqual(sequence["schema_version"], "vi_pose_v1")
            self.assertEqual(sequence["dataset"], "euroc")
            self.assertEqual(sequence["sequence_name"], "MH_01_easy")
            self.assertEqual(sequence["sequence_path"], "machine_hall/MH_01_easy")
            self.assertNotIn("split", sequence)
            self.assertEqual(sequence["sensor"]["T_cam_imu"], np.eye(4).tolist())
            self.assertEqual(sequence["sensor"]["T_imu_cam"], np.eye(4).tolist())
            self.assertEqual(sequence["sensor"]["distortion_model"], "radial-tangential")

            frame = sequence["frames"][1]
            self.assertEqual(frame["frame_id"], 1)
            self.assertIn("extrinsics", frame)
            self.assertIn("extrinsics_w2c", frame)
            self.assertTrue(
                np.allclose(
                    frame["extrinsics"],
                    np.array(
                        [
                            [1.0, 0.0, 0.0, -1.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                        ]
                    ),
                    atol=1e-6,
                )
            )
            self.assertNotIn("clean_image_rel_path", frame)
            self.assertNotIn("degradation", frame)
            self.assertTrue(np.allclose(frame["extrinsics"], frame["extrinsics_w2c"]))

            with open(output_dir / "sequence_manifest.json", "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["dataset"], "euroc")
            self.assertEqual(manifest["split_policy"], "configured_in_training")
            self.assertEqual(manifest["sequences"]["MH_01_easy"]["file"], "machine_hall__MH_01_easy.jgz")
            self.assertEqual(manifest["sequences"]["MH_01_easy"]["frame_count"], 2)
            self.assertEqual(manifest["sequences"]["MH_01_easy"]["camera_names"], ["cam0"])
            self.assertEqual(
                manifest["sequences"]["MH_01_easy"]["distortion_models"],
                {"cam0": "radial-tangential"},
            )

    def test_main_preserves_equidistant_distortion_for_uma_bumblebee(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            uma_dir = root / "uma"
            output_dir = root / "anno"
            _write_asl_sequence(uma_dir, "corridor_01")
            sensor_path = uma_dir / "corridor_01" / "mav0" / "cam0" / "sensor.yaml"
            with open(sensor_path, "r", encoding="utf-8") as f:
                sensor = yaml.safe_load(f)
            sensor["distortion_model"] = "equidistant"
            with open(sensor_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(sensor, f, sort_keys=False)

            argv = [
                "generate_euroc_annotations.py",
                "--asl_dir",
                str(uma_dir),
                "--output_dir",
                str(output_dir),
                "--dataset_name",
                "uma_vi",
            ]
            with mock.patch.object(sys, "argv", argv):
                gen_euroc.main()

            with gzip.open(output_dir / "corridor_01.jgz", "rt", encoding="utf-8") as f:
                payload = json.load(f)
            sequence = payload["uma_vi/corridor_01/cam0"]

            self.assertEqual(sequence["dataset"], "uma_vi")
            self.assertEqual(sequence["sensor"]["distortion_model"], "equidistant")

            with open(output_dir / "sequence_manifest.json", "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(
                manifest["sequences"]["corridor_01"]["distortion_models"],
                {"cam0": "equidistant"},
            )

    def test_inspect_vi_dataset_writes_report_and_sequence_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            euroc_dir = root / "euroc"
            output_dir = root / "anno"
            _write_asl_sequence(euroc_dir, "machine_hall/MH_01_easy")
            _write_asl_sequence(euroc_dir, "machine_hall/MH_05_difficult")
            _write_asl_sequence(euroc_dir, "vicon_room1/V1_01_easy")
            _write_asl_sequence(euroc_dir, "vicon_room2/V2_02_medium")

            argv = [
                "inspect_vi_dataset.py",
                "--dataset",
                "euroc",
                "--data_root",
                str(euroc_dir),
                "--output",
                str(output_dir / "inspect_report.json"),
            ]
            with mock.patch.object(sys, "argv", argv):
                inspect_vi_dataset.main()

            with open(output_dir / "inspect_report.json", "r", encoding="utf-8") as f:
                report = json.load(f)
            with open(output_dir / "sequence_manifest.json", "r", encoding="utf-8") as f:
                manifest = json.load(f)

            self.assertEqual(report["sequence_count"], 4)
            self.assertEqual(report["sequences"]["MH_01_easy"]["gt_count"], 2)
            self.assertEqual(report["sequences"]["MH_01_easy"]["imu_count"], 1)
            self.assertEqual(manifest["split_policy"], "configured_in_training")
            self.assertEqual(manifest["sequences"]["MH_01_easy"]["sequence_path"], "machine_hall/MH_01_easy")
            self.assertEqual(manifest["sequences"]["MH_01_easy"]["frame_count"], 2)


if __name__ == "__main__":
    unittest.main()
