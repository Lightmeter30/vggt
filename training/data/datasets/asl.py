# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import logging
import os.path as osp
import random
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from data.base_dataset import BaseDataset
from data.dataset_util import *


class ASLDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        dataset_name: str = "euroc",
        ASL_DIR: str = None,
        ASL_ANNOTATION_DIR: str = None,
        EUROC_DIR: str = None,
        EUROC_ANNOTATION_DIR: str = None,
        annotation_prefix: str = None,
        annotation_mode: str = "per_sequence",
        sequence_names: Optional[Sequence[str]] = None,
        min_num_images: int = 24,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        camera_names: Sequence[str] = ("cam0",),
        train_split_ratio: float = 0.8,
        undistort_images: bool = True,
        max_pose_time_diff_ns: int = 10_000_000,
        load_imu: bool = False,
        imu_window_ns: int = 100_000_000,
        imu_num_samples: int = 32,
        imu_feature_order: str = "gyro_accel",
        empty_imu_policy: str = "zeros",
    ):
        """
        Dataset wrapper for EuRoC-style ASL/MAV visual-inertial sequences.

        The heavy preprocessing steps (sequence discovery, GT alignment,
        calibration parsing, and IMU export) are handled offline by the ASL
        annotation generator. Train/val/test membership is intentionally supplied
        by config through sequence_names.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        if ASL_DIR is None:
            ASL_DIR = EUROC_DIR
        if ASL_ANNOTATION_DIR is None:
            ASL_ANNOTATION_DIR = EUROC_ANNOTATION_DIR
        if ASL_DIR is None or ASL_ANNOTATION_DIR is None:
            raise ValueError("Both ASL_DIR and ASL_ANNOTATION_DIR must be specified.")

        self.dataset_name = str(dataset_name)
        self.split = str(split)
        self.annotation_prefix = annotation_prefix or self.dataset_name
        self.annotation_mode = str(annotation_mode)
        self.sequence_names = None if sequence_names is None else tuple(sequence_names)
        self.ASL_DIR = osp.abspath(ASL_DIR)
        self.ASL_ANNOTATION_DIR = osp.abspath(ASL_ANNOTATION_DIR)
        self.EUROC_DIR = self.ASL_DIR
        self.EUROC_ANNOTATION_DIR = self.ASL_ANNOTATION_DIR
        self.min_num_images = min_num_images
        self.expand_ratio = expand_ratio
        self.camera_names = tuple(camera_names)
        self.undistort_images = undistort_images
        self.load_imu = load_imu
        self.imu_window_ns = int(imu_window_ns)
        self.imu_num_samples = int(imu_num_samples)
        self.imu_feature_order = imu_feature_order
        self.empty_imu_policy = empty_imu_policy
        if self.imu_num_samples <= 0:
            raise ValueError(f"imu_num_samples must be positive, got {imu_num_samples}")
        if self.imu_feature_order not in ("gyro_accel", "accel_gyro"):
            raise ValueError(f"Unsupported imu_feature_order: {imu_feature_order}")
        if self.empty_imu_policy != "zeros":
            raise ValueError(f"Unsupported empty_imu_policy: {empty_imu_policy}")
        self.epoch = 0

        # These knobs now belong to the offline preprocessing step. Keep them in
        # the signature for config compatibility.
        self.train_split_ratio = train_split_ratio
        self.max_pose_time_diff_ns = int(max_pose_time_diff_ns)

        explicit_empty = self.sequence_names is not None and len(self.sequence_names) == 0
        if not osp.isdir(self.ASL_DIR) and not explicit_empty:
            raise ValueError(f"ASL_DIR does not exist: {self.ASL_DIR}")
        if not osp.isdir(self.ASL_ANNOTATION_DIR) and not explicit_empty:
            raise ValueError(
                f"ASL_ANNOTATION_DIR does not exist: {self.ASL_ANNOTATION_DIR}"
            )

        if split == "train":
            self.len_train = len_train
        elif split in ("val", "test"):
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        annotation = self._load_annotation(split)

        self.data_store: Dict[str, Dict] = {}
        total_frame_num = 0

        for sequence_name, raw_sequence_data in sorted(annotation.items()):
            sequence_data = self._deserialize_sequence(raw_sequence_data)
            if self.camera_names and sequence_data["camera_name"] not in self.camera_names:
                continue

            frame_count = len(sequence_data["frames"])
            if frame_count < self.min_num_images:
                logging.info(
                    "Skipping ASL sequence %s: only %d matched frames",
                    sequence_name,
                    frame_count,
                )
                continue

            self.data_store[sequence_name] = sequence_data
            total_frame_num += frame_count

        self.sequence_list = sorted(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num
        if self.sequence_list_len == 0:
            self.len_train = 0

        if self.debug and self.sequence_list:
            self.sequence_list = self.sequence_list[:1]
            self.sequence_list_len = len(self.sequence_list)

        if split in ("val", "test"):
            self.len_train = min(self.len_train, self.sequence_list_len)

        status = "Training" if self.training else "Testing"
        logging.info("ASL_DIR is %s", self.ASL_DIR)
        logging.info("ASL_ANNOTATION_DIR is %s", self.ASL_ANNOTATION_DIR)
        logging.info("%s: %s sequence count: %d", status, self.dataset_name, self.sequence_list_len)
        logging.info("%s: %s matched frame count: %d", status, self.dataset_name, self.total_frame_num)
        logging.info("%s: %s dataset length: %d", status, self.dataset_name, len(self))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _load_annotation(self, split: str) -> Dict[str, Dict]:
        if self.sequence_names is not None and len(self.sequence_names) == 0:
            return {}
        if self.annotation_mode == "legacy_split":
            return self._load_split_annotation(split)

        manifest_path = osp.join(self.ASL_ANNOTATION_DIR, "sequence_manifest.json")
        if not osp.isfile(manifest_path):
            return self._load_split_annotation(split)

        with open(manifest_path, "r", encoding="utf-8") as fin:
            manifest = json.load(fin)

        manifest_sequences = manifest.get("sequences", {})
        requested = list(self.sequence_names) if self.sequence_names is not None else sorted(manifest_sequences)
        annotation: Dict[str, Dict] = {}
        for sequence_name in requested:
            manifest_key = self._resolve_manifest_sequence_key(sequence_name, manifest_sequences)
            record = manifest_sequences[manifest_key]
            annotation_file = osp.join(self.ASL_ANNOTATION_DIR, record["file"])
            if not osp.isfile(annotation_file):
                raise FileNotFoundError(f"ASL sequence annotation file not found: {annotation_file}")
            with gzip.open(annotation_file, "rt", encoding="utf-8") as fin:
                annotation.update(json.load(fin))
        return annotation

    def _resolve_manifest_sequence_key(self, sequence_name: str, manifest_sequences: Dict[str, Dict]) -> str:
        if sequence_name in manifest_sequences:
            return sequence_name
        short_name = str(sequence_name).rstrip("/").split("/")[-1]
        if short_name in manifest_sequences:
            return short_name
        matches = [
            name
            for name, record in manifest_sequences.items()
            if record.get("sequence_path") == sequence_name
        ]
        if len(matches) == 1:
            return matches[0]
        raise KeyError(f"Sequence {sequence_name} not found in ASL sequence_manifest.json")

    def _load_split_annotation(self, split: str) -> Dict[str, Dict]:
        annotation_file = osp.join(self.ASL_ANNOTATION_DIR, f"{self.annotation_prefix}_{split}.jgz")
        if not osp.isfile(annotation_file):
            raise FileNotFoundError(
                f"ASL annotation file not found: {annotation_file}. "
                "Run training/data/preprocess/generate_euroc_annotations.py first."
            )

        with gzip.open(annotation_file, "rt", encoding="utf-8") as fin:
            return json.load(fin)

    def _deserialize_sequence(self, sequence_data: Dict) -> Dict:
        sensor = sequence_data["sensor"]
        sensor = {
            "intrinsics": np.asarray(sensor["intrinsics"], dtype=np.float32),
            "distortion": np.asarray(sensor["distortion"], dtype=np.float32),
            "undistorted_intrinsics": np.asarray(
                sensor["undistorted_intrinsics"], dtype=np.float32
            ),
            "image_size": np.asarray(sensor["image_size"], dtype=np.int32),
            "distortion_model": str(sensor.get("distortion_model", "radial-tangential")),
        }

        frames = []
        for frame in sequence_data["frames"]:
            frames.append(
                {
                    "frame_id": int(frame.get("frame_id", len(frames))),
                    "timestamp_ns": int(frame["timestamp_ns"]),
                    "gt_timestamp_ns": int(frame["gt_timestamp_ns"]),
                    "pose_dt_ns": int(frame["pose_dt_ns"]),
                    "image_rel_path": frame["image_rel_path"],
                    "extrinsics": np.asarray(
                        frame.get("extrinsics", frame.get("extrinsics_w2c")),
                        dtype=np.float32,
                    ),
                }
            )

        imu_data = sequence_data.get("imu_data")
        if imu_data is not None:
            imu_data = {
                "timestamps_ns": np.asarray(imu_data["timestamps_ns"], dtype=np.int64),
                "gyro": np.asarray(imu_data["gyro"], dtype=np.float32),
                "accel": np.asarray(imu_data["accel"], dtype=np.float32),
            }

        return {
            "camera_name": sequence_data["camera_name"],
            "dataset": sequence_data.get("dataset", self.dataset_name),
            "sequence_name": sequence_data.get("sequence_name"),
            "sequence_path": sequence_data.get("sequence_path"),
            "split": self.split,
            "sensor": sensor,
            "frames": frames,
            "imu_data": imu_data,
        }

    def _undistort_image(
        self,
        image: np.ndarray,
        sensor: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.undistort_images:
            return image, sensor["intrinsics"].copy()

        if sensor.get("distortion_model") == "equidistant":
            distortion = sensor["distortion"].reshape(-1, 1)
            undistorted = cv2.fisheye.undistortImage(
                image,
                sensor["intrinsics"],
                distortion,
                Knew=sensor["undistorted_intrinsics"],
            )
            return undistorted, sensor["undistorted_intrinsics"].copy()

        undistorted = cv2.undistort(
            image,
            sensor["intrinsics"],
            sensor["distortion"],
            None,
            sensor["undistorted_intrinsics"],
        )
        return undistorted, sensor["undistorted_intrinsics"].copy()

    def _build_placeholder_depth(
        self, image_shape: Tuple[int, int], intrinsics: np.ndarray
    ) -> np.ndarray:
        height, width = image_shape[:2]
        depth_map = np.zeros((height, width), dtype=np.float32)

        u = int(np.clip(np.round(intrinsics[0, 2]), 0, width - 1))
        v = int(np.clip(np.round(intrinsics[1, 2]), 0, height - 1))
        depth_map[v, u] = 1.0
        return depth_map

    def _sparsify_placeholder_geometry(
        self,
        depth_map: np.ndarray,
        cam_coords_points: np.ndarray,
        world_coords_points: np.ndarray,
        point_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sparse_mask = np.zeros_like(point_mask, dtype=bool)
        valid_coords = np.argwhere(point_mask)

        if len(valid_coords) > 0:
            row, col = valid_coords[len(valid_coords) // 2]
            sparse_mask[row, col] = True

        sparse_depth = np.where(sparse_mask, depth_map, 0.0).astype(np.float32)
        sparse_cam_points = np.where(
            sparse_mask[..., None], cam_coords_points, 0.0
        ).astype(np.float32)
        sparse_world_points = np.where(
            sparse_mask[..., None], world_coords_points, 0.0
        ).astype(np.float32)
        return sparse_depth, sparse_cam_points, sparse_world_points, sparse_mask

    def _empty_imu_window(self) -> Tuple[np.ndarray, np.ndarray]:
        window = np.zeros((self.imu_num_samples, 6), dtype=np.float32)
        mask = np.zeros((self.imu_num_samples,), dtype=bool)
        return window, mask

    def _format_imu_features(
        self, gyro: np.ndarray, accel: np.ndarray
    ) -> np.ndarray:
        if self.imu_feature_order == "gyro_accel":
            return np.concatenate([gyro, accel], axis=-1).astype(np.float32)
        return np.concatenate([accel, gyro], axis=-1).astype(np.float32)

    def _load_imu_window(
        self, imu_data: Optional[Dict[str, np.ndarray]], center_timestamp_ns: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if imu_data is None:
            logging.warning(
                "EuRoC IMU data is missing; returning zero IMU window."
            )
            return self._empty_imu_window()

        timestamps = imu_data["timestamps_ns"]
        if len(timestamps) == 0:
            logging.warning(
                "EuRoC IMU data is empty; returning zero IMU window."
            )
            return self._empty_imu_window()

        start_ts = center_timestamp_ns - self.imu_window_ns
        end_ts = center_timestamp_ns + self.imu_window_ns
        left = int(np.searchsorted(timestamps, start_ts, side="left"))
        right = int(np.searchsorted(timestamps, end_ts, side="right"))

        window_timestamps = timestamps[left:right]
        if len(window_timestamps) == 0:
            return self._empty_imu_window()

        target_timestamps = np.linspace(
            start_ts,
            end_ts,
            num=self.imu_num_samples,
            dtype=np.float64,
        )
        gyro = np.zeros((self.imu_num_samples, 3), dtype=np.float32)
        accel = np.zeros((self.imu_num_samples, 3), dtype=np.float32)
        valid_mask = (
            (target_timestamps >= float(window_timestamps[0]))
            & (target_timestamps <= float(window_timestamps[-1]))
        )

        for axis in range(3):
            gyro[:, axis] = np.interp(
                target_timestamps,
                window_timestamps.astype(np.float64),
                imu_data["gyro"][left:right, axis].astype(np.float64),
            ).astype(np.float32)
            accel[:, axis] = np.interp(
                target_timestamps,
                window_timestamps.astype(np.float64),
                imu_data["accel"][left:right, axis].astype(np.float64),
            ).astype(np.float32)

        imu_window = self._format_imu_features(gyro, accel)
        imu_window[~valid_mask] = 0.0
        return imu_window, valid_mask.astype(bool)

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if self.sequence_list_len == 0:
            raise RuntimeError("No valid ASL sequences were loaded.")

        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        sequence_data = self.data_store[seq_name]
        frames = sequence_data["frames"]
        sensor = sequence_data["sensor"]
        num_images = len(frames)

        if ids is None:
            ids = np.random.choice(
                num_images, img_per_seq, replace=self.allow_duplicate_img
            )

        if self.get_nearby:
            ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []
        imu_windows = []
        imu_window_masks = []
        timestamps_ns = []

        for image_idx in ids:
            frame = frames[int(image_idx)]
            image_path = osp.join(self.ASL_DIR, frame["image_rel_path"])
            image = read_image_cv2(image_path)
            if image is None:
                logging.warning("Failed to read ASL image: %s", image_path)
                continue

            image, intri_opencv = self._undistort_image(image, sensor)

            original_size = np.array(image.shape[:2])
            extri_opencv = frame["extrinsics"].copy()
            depth_map = self._build_placeholder_depth(image.shape[:2], intri_opencv)

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=image_path,
            )

            depth_map, cam_coords_points, world_coords_points, point_mask = (
                self._sparsify_placeholder_geometry(
                    depth_map,
                    cam_coords_points,
                    world_coords_points,
                    point_mask,
                )
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

            if self.load_imu:
                imu_window, imu_window_mask = self._load_imu_window(
                    sequence_data["imu_data"], frame["timestamp_ns"]
                )
                imu_windows.append(imu_window)
                imu_window_masks.append(imu_window_mask)
                timestamps_ns.append(int(frame["timestamp_ns"]))

        batch = {
            "seq_name": f"{self.dataset_name}_" + seq_name,
            "ids": np.asarray(ids, dtype=np.int64),
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }

        if self.load_imu:
            batch["imu_windows"] = np.stack(imu_windows, axis=0).astype(np.float32)
            batch["imu_window_masks"] = np.stack(imu_window_masks, axis=0).astype(bool)
            batch["timestamps_ns"] = np.asarray(timestamps_ns, dtype=np.int64)

        return batch
