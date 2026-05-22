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

from training.data.preprocess import generate_tum_vi_annotations as gen_tum_vi


def _write_csv(csv_path: Path, header, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_tum_vi_sequence(root: Path, sequence_name: str):
    sequence_root = root / sequence_name
    cam0_dir = sequence_root / "mav0" / "cam0"
    mocap_dir = sequence_root / "mav0" / "mocap0"
    imu0_dir = sequence_root / "mav0" / "imu0"
    dso_dir = sequence_root / "dso"

    timestamps = [1_000_000_000, 1_050_000_000]
    cam_rows = []
    for idx, timestamp in enumerate(timestamps):
        image = np.full((16, 16, 3), 40 + idx, dtype=np.uint8)
        image_name = f"{timestamp}.png"
        image_path = cam0_dir / "data" / image_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(image_path), image)
        cam_rows.append([timestamp, image_name])
    _write_csv(cam0_dir / "data.csv", ["#timestamp [ns]", "filename"], cam_rows)

    mocap_rows = [
        [timestamps[0], 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [timestamps[1], 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ]
    _write_csv(
        mocap_dir / "data.csv",
        [
            "#timestamp [ns]",
            "p_RS_R_x [m]",
            "p_RS_R_y [m]",
            "p_RS_R_z [m]",
            "q_RS_w []",
            "q_RS_x []",
            "q_RS_y []",
            "q_RS_z []",
        ],
        mocap_rows,
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

    camera_from_imu = np.eye(4, dtype=np.float32)
    camera_from_imu[0, 3] = 0.2
    camchain = {
        "cam0": {
            "camera_model": "pinhole",
            "distortion_model": "equidistant",
            "distortion_coeffs": [0.01, -0.02, 0.001, 0.0005],
            "intrinsics": [12.0, 12.0, 8.0, 8.0],
            "resolution": [16, 16],
            "T_cam_imu": camera_from_imu.tolist(),
        }
    }
    dso_dir.mkdir(parents=True, exist_ok=True)
    with open(dso_dir / "camchain.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(camchain, f, sort_keys=False)


class TestGenerateTumViAnnotations(unittest.TestCase):
    def test_main_writes_sequence_jgz_using_t_cam_imu_as_camera_from_imu(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tum_vi_dir = root / "tum"
            output_dir = root / "anno"
            _write_tum_vi_sequence(tum_vi_dir, "dataset-room1_512_16")

            argv = [
                "generate_tum_vi_annotations.py",
                "--tum_vi_dir",
                str(tum_vi_dir),
                "--output_dir",
                str(output_dir),
                "--camera_names",
                "cam0",
            ]
            with mock.patch.object(sys, "argv", argv):
                gen_tum_vi.main()

            output_file = output_dir / "dataset-room1_512_16.jgz"
            self.assertTrue(output_file.is_file())

            with gzip.open(output_file, "rt", encoding="utf-8") as f:
                payload = json.load(f)

            sequence = payload["dataset-room1_512_16"]
            self.assertEqual(sequence["sensor"]["distortion_model"], "equidistant")
            self.assertEqual(sequence["sensor"]["intrinsics"][0][0], 12.0)
            self.assertIn("undistorted_intrinsics", sequence["sensor"])

            frame = sequence["frames"][1]
            self.assertIn("extrinsics_w2c", frame)
            self.assertNotIn("extrinsics", frame)
            self.assertTrue(
                np.allclose(
                    frame["extrinsics_w2c"],
                    np.array(
                        [
                            [1.0, 0.0, 0.0, -0.8],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                        ]
                    ),
                    atol=1e-6,
                )
            )

            with open(output_dir / "summary.json", "r", encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary["dataset_format"], "tum_vi")
            self.assertEqual(summary["generated_files"], ["dataset-room1_512_16.jgz"])


if __name__ == "__main__":
    unittest.main()
