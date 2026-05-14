import gzip
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from vggt.utils.load_fn import load_and_preprocess_images_from_objects

from evaluation.common.model import _extract_state_dict, _strip_state_dict_prefixes
from evaluation.common.metrics import calculate_auc_np, se3_to_relative_pose_error
from evaluation.datasets.euroc.camera_pose import (
    evaluate_sequences,
    load_euroc_sequence_entries,
)
from evaluation.datasets.realestate10k.camera_pose import (
    evaluate_sequences as evaluate_realestate10k_sequences,
    load_frame_manifest,
    load_realestate10k_sequence_entries,
    write_metrics_report,
)
from training.data.preprocess.generate_local_realestate10k_frames import (
    find_best_metadata_match,
    frame_tolerance_us,
)
from evaluation.datasets.co3d.camera_pose import (
    evaluate_sequences as evaluate_co3d_sequences,
    load_co3d_sequence_entries,
)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[..., 0] = color[0]
    image[..., 1] = color[1]
    image[..., 2] = color[2]
    assert cv2.imwrite(str(path), image)


def _make_extrinsic(tx: float) -> list[list[float]]:
    extrinsic = np.hstack([np.eye(3, dtype=np.float32), np.array([[tx], [0.0], [0.0]], dtype=np.float32)])
    return extrinsic.tolist()


def _make_sequence_payload(camera_name: str, frame_paths: list[str], tx_values: list[float]) -> dict:
    return {
        "camera_name": camera_name,
        "sensor": {
            "intrinsics": [[40.0, 0.0, 16.0], [0.0, 40.0, 16.0], [0.0, 0.0, 1.0]],
            "distortion": [0.0, 0.0, 0.0, 0.0],
            "undistorted_intrinsics": [[40.0, 0.0, 16.0], [0.0, 40.0, 16.0], [0.0, 0.0, 1.0]],
            "image_size": [32, 32],
        },
        "frames": [
            {
                "timestamp_ns": idx + 1,
                "gt_timestamp_ns": idx + 1,
                "pose_dt_ns": 0,
                "image_rel_path": frame_path,
                "extrinsics": _make_extrinsic(tx),
            }
            for idx, (frame_path, tx) in enumerate(zip(frame_paths, tx_values))
        ],
        "imu_data": None,
    }


class TestLoadAndPreprocessImagesFromObjects(unittest.TestCase):
    def test_accepts_pil_images_and_numpy_arrays(self):
        pil_image = Image.new("RGB", (20, 10), color=(255, 0, 0))
        rgba_array = np.zeros((12, 18, 4), dtype=np.uint8)
        rgba_array[..., 1] = 255
        rgba_array[..., 3] = 255

        images = load_and_preprocess_images_from_objects([pil_image, rgba_array], mode="pad")

        self.assertEqual(images.shape, (2, 3, 518, 518))
        self.assertTrue(torch.isfinite(images).all())


class TestEvaluationMetrics(unittest.TestCase):
    def test_identical_relative_poses_have_zero_error_and_perfect_auc(self):
        extrinsics = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0, 2.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            ],
            dtype=torch.float64,
        )
        add_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64).expand(3, 1, 4)
        se3 = torch.cat([extrinsics, add_row], dim=1)

        r_error, t_error = se3_to_relative_pose_error(se3, se3, num_frames=3)
        auc_30, _ = calculate_auc_np(r_error.cpu().numpy(), t_error.cpu().numpy(), max_threshold=30)

        self.assertTrue(torch.allclose(r_error, torch.zeros_like(r_error), atol=1e-5))
        self.assertTrue(torch.allclose(t_error, torch.zeros_like(t_error), atol=1e-5))
        self.assertAlmostEqual(auc_30, 1.0)


class TestEvaluationModelLoading(unittest.TestCase):
    def test_extract_state_dict_accepts_nested_training_checkpoint(self):
        tensor = torch.ones(1)
        state_dict = _extract_state_dict({"model": {"module.camera_head.weight": tensor}})
        cleaned = _strip_state_dict_prefixes(state_dict)

        self.assertEqual(list(cleaned.keys()), ["camera_head.weight"])
        self.assertTrue(torch.equal(cleaned["camera_head.weight"], tensor))


class TestEurocCameraPoseEvaluation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.euroc_dir = self.root / "euroc"
        self.anno_dir = self.root / "anno"
        frame_paths = [
            "machine_hall/MH_01_easy/mav0/cam0/data/1.png",
            "machine_hall/MH_01_easy/mav0/cam0/data/2.png",
            "machine_hall/MH_01_easy/mav0/cam0/data/3.png",
        ]
        for idx, frame_path in enumerate(frame_paths):
            _write_image(self.euroc_dir / frame_path, color=(idx * 10, 20, 30))

        annotation = {
            "machine_hall/MH_01_easy:cam0": _make_sequence_payload("cam0", frame_paths, [0.0, 1.0, 2.0]),
            "machine_hall/MH_01_easy:cam1": _make_sequence_payload("cam1", frame_paths, [0.0, 1.1, 2.2]),
            "machine_hall/MH_short:cam0": _make_sequence_payload("cam0", frame_paths[:1], [0.0]),
        }
        self.anno_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.anno_dir / "euroc_test.jgz", "wt", encoding="utf-8") as fout:
            json.dump(annotation, fout)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_euroc_sequence_entries_filters_camera_and_frame_count(self):
        sequence_entries = load_euroc_sequence_entries(
            annotation_path=self.anno_dir / "euroc_test.jgz",
            euroc_dir=self.euroc_dir,
            camera_names=("cam0",),
            min_num_images=2,
        )

        self.assertEqual(len(sequence_entries), 1)
        self.assertEqual(sequence_entries[0]["seq_name"], "machine_hall/MH_01_easy:cam0")
        self.assertEqual(len(sequence_entries[0]["frames"]), 3)

    def test_evaluate_sequences_returns_perfect_metrics_with_gt_predictor(self):
        sequence_entries = load_euroc_sequence_entries(
            annotation_path=self.anno_dir / "euroc_test.jgz",
            euroc_dir=self.euroc_dir,
            camera_names=("cam0",),
            min_num_images=2,
        )

        def predictor(model, image_paths, image_objects, frame_entries, device, dtype):
            del model, image_paths, image_objects, device, dtype
            return torch.from_numpy(
                np.stack([frame["extrinsics"] for frame in frame_entries], axis=0)
            ).to(torch.float64)

        result = evaluate_sequences(
            model=None,
            sequence_entries=sequence_entries,
            num_frames=3,
            fast_eval=False,
            seed=0,
            device="cpu",
            dtype=torch.float32,
            undistort_images=True,
            predictor=predictor,
        )

        self.assertEqual(result["num_sequences"], 1)
        self.assertAlmostEqual(result["AUC@30"], 1.0)
        self.assertAlmostEqual(result["AUC@15"], 1.0)
        self.assertAlmostEqual(result["AUC@5"], 1.0)
        self.assertAlmostEqual(result["AUC@3"], 1.0)


class TestRealEstate10KCameraPoseEvaluation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dataset_dir = self.root / "realEstate10K"
        self.split_dir = self.dataset_dir / "test"
        self.transcode_dir = self.dataset_dir / "transcode"
        self.split_dir.mkdir(parents=True, exist_ok=True)

        video_id = "abc123XYZ"
        frame_timestamps = ["1000", "2000", "3000"]
        image_dir = self.transcode_dir / video_id
        for idx, timestamp in enumerate(frame_timestamps):
            _write_image(image_dir / f"{timestamp}.jpg", color=(idx * 20, 40, 80))

        self._write_realestate10k_txt(
            self.split_dir / "valid.txt",
            video_id=video_id,
            frame_timestamps=frame_timestamps,
            tx_values=[0.0, 1.0, 2.0],
        )
        self._write_realestate10k_txt(
            self.split_dir / "expired.txt",
            video_id="missingVIDEO",
            frame_timestamps=["1000", "2000", "3000"],
            tx_values=[0.0, 1.0, 2.0],
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_realestate10k_txt(self, path: Path, video_id: str, frame_timestamps: list[str], tx_values: list[float]):
        rows = [f"https://www.youtube.com/watch?v={video_id}"]
        for timestamp, tx in zip(frame_timestamps, tx_values):
            extrinsic = np.asarray(_make_extrinsic(tx), dtype=np.float64).reshape(-1)
            values = [
                timestamp,
                "0.75",
                "1.25",
                "0.5",
                "0.5",
                "0.0",
                "0.0",
            ]
            values.extend(f"{value:.9f}" for value in extrinsic)
            rows.append(" ".join(values))
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def test_load_realestate10k_sequence_entries_skips_missing_videos(self):
        sequence_entries = load_realestate10k_sequence_entries(
            realestate10k_dir=self.dataset_dir,
            split="test",
            min_num_images=2,
        )

        self.assertEqual(len(sequence_entries), 1)
        self.assertEqual(sequence_entries[0]["seq_name"], "valid")
        self.assertEqual(sequence_entries[0]["video_id"], "abc123XYZ")
        self.assertEqual(len(sequence_entries[0]["frames"]), 3)
        self.assertEqual(sequence_entries[0]["frames"][0]["timestamp"], "1000")
        self.assertEqual(sequence_entries[0]["frames"][0]["image_path"], str(self.transcode_dir / "abc123XYZ" / "1000.jpg"))
        self.assertTrue(np.allclose(sequence_entries[0]["frames"][1]["extrinsics"], _make_extrinsic(1.0)))
        self.assertTrue(np.isclose(sequence_entries[0]["frames"][0]["intrinsics_normalized"][0], 0.75))

    def test_load_realestate10k_sequence_entries_rejects_stale_manifest_frames(self):
        manifest_path = self.dataset_dir / "transcode_manifest.jsonl"
        manifest_rows = [
            {
                "seq_name": "valid",
                "video_id": "abc123XYZ",
                "timestamp": "1000",
                "matched_video_time_us": 1000,
                "abs_error_us": 0.0,
                "fps": 30.0,
                "image_path": str(self.transcode_dir / "abc123XYZ" / "1000.jpg"),
                "source_video_path": str(self.dataset_dir / "downloaded" / "abc123XYZ"),
            },
            {
                "seq_name": "valid",
                "video_id": "abc123XYZ",
                "timestamp": "2000",
                "matched_video_time_us": 2000,
                "abs_error_us": 20000.0,
                "fps": 30.0,
                "image_path": str(self.transcode_dir / "abc123XYZ" / "2000.jpg"),
                "source_video_path": str(self.dataset_dir / "downloaded" / "abc123XYZ"),
            },
            {
                "seq_name": "valid",
                "video_id": "abc123XYZ",
                "timestamp": "3000",
                "matched_video_time_us": 3000,
                "abs_error_us": 0.0,
                "fps": 30.0,
                "image_path": str(self.transcode_dir / "abc123XYZ" / "3000.jpg"),
                "source_video_path": str(self.dataset_dir / "downloaded" / "abc123XYZ"),
            },
        ]
        manifest_path.write_text(
            "\n".join(json.dumps(row) for row in manifest_rows) + "\n",
            encoding="utf-8",
        )

        sequence_entries = load_realestate10k_sequence_entries(
            realestate10k_dir=self.dataset_dir,
            split="test",
            min_num_images=2,
            frame_manifest=load_frame_manifest(manifest_path),
            require_frame_manifest=True,
        )

        self.assertEqual(len(sequence_entries), 1)
        self.assertEqual([frame["timestamp"] for frame in sequence_entries[0]["frames"]], ["1000", "3000"])
        self.assertEqual(sequence_entries[0]["frame_filter_stats"]["stale_manifest"], 1)

    def test_evaluate_sequences_returns_perfect_metrics_with_gt_predictor(self):
        sequence_entries = load_realestate10k_sequence_entries(
            realestate10k_dir=self.dataset_dir,
            split="test",
            min_num_images=2,
        )

        def predictor(model, image_paths, image_objects, frame_entries, device, dtype, preprocess_mode="crop"):
            del model, image_paths, image_objects, device, dtype, preprocess_mode
            return torch.from_numpy(
                np.stack([frame["extrinsics"] for frame in frame_entries], axis=0)
            ).to(torch.float64)

        result = evaluate_realestate10k_sequences(
            model=None,
            sequence_entries=sequence_entries,
            num_frames=3,
            fast_eval=False,
            max_sequences=None,
            seed=0,
            device="cpu",
            dtype=torch.float32,
            thresholds=(3, 5, 15, 30),
            preprocess_mode="crop",
            predictor=predictor,
        )

        self.assertEqual(result["num_sequences"], 1)
        for threshold in (3, 5, 15, 30):
            self.assertAlmostEqual(result[f"RRA@{threshold}"], 1.0)
            self.assertAlmostEqual(result[f"RTA@{threshold}"], 1.0)
            self.assertAlmostEqual(result[f"AUC@{threshold}"], 1.0)

    def test_fast_eval_samples_sequences_before_min_frame_filter_like_co3d(self):
        sequence_entries = []
        for idx in range(11):
            frame_count = 3 if idx == 8 else 1
            sequence_entries.append(
                {
                    "seq_name": f"seq{idx:02d}",
                    "video_id": f"video{idx:02d}",
                    "frames": [
                        {
                            "image_path": str(self.transcode_dir / "abc123XYZ" / "1000.jpg"),
                            "extrinsics": _make_extrinsic(float(frame_idx)),
                        }
                        for frame_idx in range(frame_count)
                    ],
                }
            )

        result = evaluate_realestate10k_sequences(
            model=None,
            sequence_entries=sequence_entries,
            num_frames=3,
            fast_eval=True,
            max_sequences=None,
            seed=0,
            device="cpu",
            dtype=torch.float32,
            thresholds=(3,),
            preprocess_mode="crop",
            min_num_images=3,
            predictor=None,
        )

        self.assertEqual(result["num_sequences"], 0)

    def test_write_metrics_report_records_summary_and_per_sequence_metrics(self):
        output_path = self.root / "metrics.txt"
        results = {
            "num_sequences": 1,
            "RRA@3": 1.0,
            "RTA@3": 0.5,
            "AUC@3": 0.75,
            "per_sequence": [
                {
                    "seq_name": "valid",
                    "video_id": "abc123XYZ",
                    "frame_indices": [0, 1, 2],
                    "rError": np.asarray([0.0, 1.0, 2.0]),
                    "tError": np.asarray([0.0, 10.0, 20.0]),
                    "RRA@3": 1.0,
                    "RTA@3": 0.3333333333,
                    "AUC@3": 0.5,
                }
            ],
        }

        write_metrics_report(
            output_path=output_path,
            results=results,
            thresholds=(3,),
            run_config={"split": "test", "num_frames": 3, "fast_eval": True},
        )

        report = output_path.read_text(encoding="utf-8")
        self.assertIn("RealEstate10K camera pose evaluation", report)
        self.assertIn("split: test", report)
        self.assertIn("num_sequences: 1", report)
        self.assertIn("RRA@3: 1.0000", report)
        self.assertIn("RTA@3: 0.5000", report)
        self.assertIn("AUC@3: 0.7500", report)
        self.assertIn("valid (abc123XYZ)", report)
        self.assertIn("frame_indices: [0, 1, 2]", report)
        self.assertIn("mean_t_error_deg: 10.0000", report)

    @unittest.skipUnless(Path("dataset/realEstate10K").exists(), "Local RealEstate10K dataset is unavailable")
    def test_loads_local_realestate10k_smoke(self):
        sequence_entries = load_realestate10k_sequence_entries(
            realestate10k_dir=Path("dataset/realEstate10K"),
            split="test",
            min_num_images=3,
        )

        self.assertGreater(len(sequence_entries), 0)
        self.assertGreaterEqual(len(sequence_entries[0]["frames"]), 3)
        for frame in sequence_entries[0]["frames"][:3]:
            self.assertTrue(Path(frame["image_path"]).exists())


class TestRealEstate10KFrameExtraction(unittest.TestCase):
    def test_frame_tolerance_uses_microseconds(self):
        self.assertAlmostEqual(frame_tolerance_us(30.0), 16666.666666666668)

    def test_find_best_metadata_match_returns_one_nearest_frame(self):
        metadata_frames = [
            {"timestamp": "30463767"},
            {"timestamp": "30497133"},
            {"timestamp": "30530500"},
        ]

        match, distance = find_best_metadata_match(
            video_time_us=30497000,
            metadata_frames=metadata_frames,
            tolerance_us=frame_tolerance_us(30.0),
            used_indices=set(),
        )

        self.assertIs(match, metadata_frames[1])
        self.assertEqual(distance, 133)


class TestCo3DCameraPoseEvaluation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.co3d_dir = self.root / "co3d"
        self.anno_dir = self.root / "anno"
        self.anno_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = [
            "apple/seq001/images/frame000001.jpg",
            "apple/seq001/images/frame000002.jpg",
            "apple/seq001/images/frame000003.jpg",
        ]
        for idx, frame_path in enumerate(frame_paths):
            _write_image(self.co3d_dir / frame_path, color=(idx * 30, 50, 90))

        annotation = {
            "seq001": [
                {
                    "filepath": frame_path,
                    "extri": _make_extrinsic(float(idx)),
                    "intri": [[40.0, 0.0, 16.0], [0.0, 40.0, 16.0], [0.0, 0.0, 1.0]],
                }
                for idx, frame_path in enumerate(frame_paths)
            ],
            "short_seq": [
                {
                    "filepath": frame_paths[0],
                    "extri": _make_extrinsic(0.0),
                    "intri": [[40.0, 0.0, 16.0], [0.0, 40.0, 16.0], [0.0, 0.0, 1.0]],
                }
            ],
        }
        with gzip.open(self.anno_dir / "apple_test.jgz", "wt", encoding="utf-8") as fout:
            json.dump(annotation, fout)
        with gzip.open(self.anno_dir / "banana_test.jgz", "wt", encoding="utf-8") as fout:
            json.dump({}, fout)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_co3d_sequence_entries_uses_local_annotation_format(self):
        sequence_entries = load_co3d_sequence_entries(
            co3d_dir=self.co3d_dir,
            co3d_anno_dir=self.anno_dir,
            split="test",
            categories=("all",),
            min_num_images=2,
        )

        self.assertEqual(len(sequence_entries), 1)
        self.assertEqual(sequence_entries[0]["category"], "apple")
        self.assertEqual(sequence_entries[0]["seq_name"], "seq001")
        self.assertEqual(len(sequence_entries[0]["frames"]), 3)
        self.assertEqual(
            sequence_entries[0]["frames"][0]["image_path"],
            str(self.co3d_dir / "apple/seq001/images/frame000001.jpg"),
        )
        self.assertTrue(
            np.allclose(sequence_entries[0]["frames"][1]["extrinsics"], _make_extrinsic(1.0))
        )

    def test_evaluate_sequences_returns_perfect_metrics_with_gt_predictor(self):
        sequence_entries = load_co3d_sequence_entries(
            co3d_dir=self.co3d_dir,
            co3d_anno_dir=self.anno_dir,
            split="test",
            categories=("apple",),
            min_num_images=2,
        )

        def predictor(model, image_paths, image_objects, frame_entries, device, dtype, preprocess_mode="crop"):
            del model, image_paths, image_objects, device, dtype, preprocess_mode
            return torch.from_numpy(
                np.stack([frame["extrinsics"] for frame in frame_entries], axis=0)
            ).to(torch.float64)

        result = evaluate_co3d_sequences(
            model=None,
            sequence_entries=sequence_entries,
            num_frames=3,
            fast_eval=False,
            max_sequences=None,
            seed=0,
            device="cpu",
            dtype=torch.float32,
            thresholds=(3, 5, 15, 30),
            preprocess_mode="crop",
            min_num_images=2,
            predictor=predictor,
        )

        self.assertEqual(result["num_sequences"], 1)
        for threshold in (3, 5, 15, 30):
            self.assertAlmostEqual(result[f"RRA@{threshold}"], 1.0)
            self.assertAlmostEqual(result[f"RTA@{threshold}"], 1.0)
            self.assertAlmostEqual(result[f"AUC@{threshold}"], 1.0)


if __name__ == "__main__":
    unittest.main()
