from pathlib import Path
import gzip
import json
import sys
import tempfile
import unittest

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from training.data.preprocess.generate_degraded_images import generate_degraded_images


def _write_annotation(annotation_path: Path, image_rel_paths):
    frames = []
    for frame_id, rel_path in enumerate(image_rel_paths):
        frames.append(
            {
                "frame_id": frame_id,
                "timestamp_ns": 1_000_000_000 + frame_id,
                "image_rel_path": rel_path,
                "clean_image_rel_path": rel_path,
                "extrinsics": np.eye(3, 4, dtype=np.float32).tolist(),
            }
        )

    payload = {
        "euroc/MH_01_easy/cam0": {
            "schema_version": "vi_annotation_v1",
            "dataset": "euroc",
            "sequence_name": "MH_01_easy",
            "sequence_path": "machine_hall/MH_01_easy",
            "camera_name": "cam0",
            "split": "val",
            "sensor": {},
            "frames": frames,
        }
    }

    with gzip.open(annotation_path, "wt", encoding="utf-8") as fout:
        json.dump(payload, fout)


def _write_clean_image(path: Path, offset: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[..., 0] = np.arange(32, dtype=np.uint8)[None, :]
    image[..., 1] = np.arange(24, dtype=np.uint8)[:, None]
    image[..., 2] = 50 + offset
    assert cv2.imwrite(str(path), image)
    return image


class TestGenerateDegradedImages(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_root = self.root / "euRoC"
        self.output_root = self.root / "degraded" / "euroc"
        self.annotation_path = self.root / "euroc_val.jgz"
        self.image_rel_paths = [
            "machine_hall/MH_01_easy/mav0/cam0/data/1000000000.png",
            "machine_hall/MH_01_easy/mav0/cam0/data/1000000001.png",
        ]
        for index, rel_path in enumerate(self.image_rel_paths):
            _write_clean_image(self.data_root / rel_path, offset=index)
        _write_annotation(self.annotation_path, self.image_rel_paths)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dry_run_does_not_write_outputs(self):
        summary = generate_degraded_images(
            annotation_path=self.annotation_path,
            data_root=self.data_root,
            output_root=self.output_root,
            settings=["exposure_medium"],
            max_frames=1,
            dry_run=True,
        )

        self.assertEqual(summary["candidate_frames"], 1)
        self.assertEqual(summary["candidate_images"], 1)
        self.assertEqual(summary["written_images"], 0)
        self.assertFalse(self.output_root.exists())

    def test_generates_images_and_jsonl_metadata_without_overwriting(self):
        summary = generate_degraded_images(
            annotation_path=self.annotation_path,
            data_root=self.data_root,
            output_root=self.output_root,
            settings=["exposure_medium"],
            max_frames=None,
            dry_run=False,
        )

        metadata_path = self.output_root / "degradation_metadata.jsonl"
        self.assertEqual(summary["candidate_frames"], 2)
        self.assertEqual(summary["candidate_images"], 2)
        self.assertEqual(summary["written_images"], 2)
        self.assertTrue(metadata_path.is_file())

        records = [
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(records), 2)
        first = records[0]
        self.assertEqual(first["schema_version"], "degradation_v1")
        self.assertEqual(first["dataset"], "euroc")
        self.assertEqual(first["sequence_name"], "MH_01_easy")
        self.assertEqual(first["camera_name"], "cam0")
        self.assertEqual(first["degradation_type"], "exposure")
        self.assertEqual(first["severity"], "medium")
        self.assertTrue(first["source_pose_unchanged"])
        self.assertTrue(first["source_intrinsics_unchanged"])
        self.assertTrue(first["source_imu_unchanged"])

        degraded_path = self.output_root / first["degraded_image_rel_path"]
        self.assertTrue(degraded_path.is_file())
        clean = cv2.imread(str(self.data_root / first["clean_image_rel_path"]), cv2.IMREAD_COLOR)
        degraded = cv2.imread(str(degraded_path), cv2.IMREAD_COLOR)
        self.assertEqual(degraded.shape, clean.shape)
        self.assertFalse(np.array_equal(degraded, clean))

        rerun_summary = generate_degraded_images(
            annotation_path=self.annotation_path,
            data_root=self.data_root,
            output_root=self.output_root,
            settings=["exposure_medium"],
            max_frames=None,
            dry_run=False,
            overwrite=False,
        )
        self.assertEqual(rerun_summary["written_images"], 0)
        self.assertEqual(rerun_summary["skipped_existing"], 2)


if __name__ == "__main__":
    unittest.main()
