import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.data.preprocess import convert_kaist_vi_bag_to_asl as convert_kaist


class TestConvertKaistViBagToAsl(unittest.TestCase):
    def test_write_sensor_yaml_converts_t_cam_imu_to_euroc_t_bs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            camera_from_imu = np.eye(4, dtype=np.float32)
            camera_from_imu[0, 3] = 0.25
            camera_config = {
                "camera_model": "pinhole",
                "distortion_model": "radtan",
                "intrinsics": [100.0, 101.0, 32.0, 33.0],
                "distortion_coeffs": [0.1, -0.2, 0.01, 0.02],
                "resolution": [640, 480],
                "T_cam_imu": camera_from_imu.tolist(),
                "rostopic": "/camera/infra1/image_rect_raw",
            }

            convert_kaist.write_camera_sensor_yaml(
                output_dir / "sensor.yaml", "cam0", camera_config
            )

            with open(output_dir / "sensor.yaml", "r", encoding="utf-8") as f:
                sensor = yaml.safe_load(f)

            self.assertEqual(sensor["sensor_type"], "camera")
            self.assertEqual(sensor["camera_model"], "pinhole")
            self.assertEqual(sensor["distortion_model"], "radial-tangential")
            self.assertEqual(sensor["intrinsics"], [100.0, 101.0, 32.0, 33.0])
            self.assertEqual(sensor["resolution"], [640, 480])
            self.assertEqual(sensor["rostopic"], "/camera/infra1/image_rect_raw")

            body_from_sensor = np.asarray(sensor["T_BS"]["data"]).reshape(4, 4)
            expected = np.eye(4, dtype=np.float32)
            expected[0, 3] = -0.25
            self.assertTrue(np.allclose(body_from_sensor, expected, atol=1e-6))

    def test_write_camera_and_imu_csvs_use_euroc_headers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            convert_kaist.write_image_csv(
                output_dir / "cam0" / "data.csv",
                [(1_000_000_000, "1000000000.png"), (1_010_000_000, "1010000000.png")],
            )
            convert_kaist.write_imu_csv(
                output_dir / "imu0" / "data.csv",
                [
                    (
                        1_000_000_000,
                        [0.1, 0.2, 0.3],
                        [1.0, 2.0, 3.0],
                    )
                ],
            )

            with open(output_dir / "cam0" / "data.csv", "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0], ["#timestamp [ns]", "filename"])
            self.assertEqual(rows[1], ["1000000000", "1000000000.png"])

            with open(output_dir / "imu0" / "data.csv", "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(
                rows[0],
                [
                    "#timestamp [ns]",
                    "w_RS_S_x [rad s^-1]",
                    "w_RS_S_y [rad s^-1]",
                    "w_RS_S_z [rad s^-1]",
                    "a_RS_S_x [m s^-2]",
                    "a_RS_S_y [m s^-2]",
                    "a_RS_S_z [m s^-2]",
                ],
            )
            self.assertEqual(rows[1], ["1000000000", "0.1", "0.2", "0.3", "1.0", "2.0", "3.0"])

    def test_write_groundtruth_csv_uses_euroc_state_format_with_zero_velocity_and_bias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            convert_kaist.write_groundtruth_csv(
                output_dir / "state_groundtruth_estimate0" / "data.csv",
                [
                    (
                        1_000_000_000,
                        [1.0, 2.0, 3.0],
                        [0.5, 0.1, 0.2, 0.3],
                    )
                ],
            )

            with open(
                output_dir / "state_groundtruth_estimate0" / "data.csv",
                "r",
                encoding="utf-8",
            ) as f:
                rows = list(csv.reader(f))

            self.assertEqual(rows[0][0], "#timestamp")
            self.assertEqual(rows[0][1:8], [
                "p_RS_R_x [m]",
                "p_RS_R_y [m]",
                "p_RS_R_z [m]",
                "q_RS_w []",
                "q_RS_x []",
                "q_RS_y []",
                "q_RS_z []",
            ])
            self.assertEqual(
                rows[1],
                [
                    "1000000000",
                    "1.0",
                    "2.0",
                    "3.0",
                    "0.5",
                    "0.1",
                    "0.2",
                    "0.3",
                    "0.0",
                    "0.0",
                    "0.0",
                    "0.0",
                    "0.0",
                    "0.0",
                    "0.0",
                    "0.0",
                    "0.0",
                ],
            )


if __name__ == "__main__":
    unittest.main()
