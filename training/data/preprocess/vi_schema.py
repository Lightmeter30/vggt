from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence


SCHEMA_VERSION = "vi_pose_v1"

SPLIT_NAMES = ("train", "val", "test")

EUROC_TRAIN_SEQUENCES = (
    "MH_01_easy",
    "MH_02_easy",
    "MH_03_medium",
    "MH_04_difficult",
    "V1_02_medium",
    "V1_03_difficult",
    "V2_01_easy",
    "V2_03_difficult",
)

EUROC_EVAL_SEQUENCES = (
    "MH_05_difficult",
    "V1_01_easy",
    "V2_02_medium",
)

EUROC_SPLIT_SEQUENCES: Dict[str, Sequence[str]] = {
    "train": EUROC_TRAIN_SEQUENCES,
    "val": EUROC_EVAL_SEQUENCES,
    "test": EUROC_EVAL_SEQUENCES,
}

EUROC_SEQUENCE_SPLITS: Dict[str, str] = {
    sequence_name: "train" for sequence_name in EUROC_TRAIN_SEQUENCES
}
EUROC_SEQUENCE_SPLITS.update(
    {sequence_name: "test" for sequence_name in EUROC_EVAL_SEQUENCES}
)


def short_sequence_name(sequence_path: str) -> str:
    return sequence_path.rstrip("/").split("/")[-1]


def sequence_entry_key(dataset: str, sequence_name: str, camera_name: str) -> str:
    return f"{dataset}/{sequence_name}/{camera_name}"


def make_empty_split_buckets() -> Dict[str, list]:
    return {split: [] for split in SPLIT_NAMES}


def normalize_split_mapping(
    split_by_sequence: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    mapping = dict(EUROC_SEQUENCE_SPLITS if split_by_sequence is None else split_by_sequence)
    invalid_splits = sorted({split for split in mapping.values() if split not in SPLIT_NAMES})
    if invalid_splits:
        raise ValueError(f"Invalid split names in manifest: {invalid_splits}")
    return mapping


def normalize_split_sequences(
    split_sequences: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, list]:
    raw_splits = EUROC_SPLIT_SEQUENCES if split_sequences is None else split_sequences
    invalid_splits = sorted({split for split in raw_splits if split not in SPLIT_NAMES})
    if invalid_splits:
        raise ValueError(f"Invalid split names in manifest: {invalid_splits}")

    normalized = make_empty_split_buckets()
    for split in SPLIT_NAMES:
        seen_in_split = set()
        for sequence_name in raw_splits.get(split, []):
            if sequence_name in seen_in_split:
                raise ValueError(f"Sequence {sequence_name} appears twice in {split}")
            normalized[split].append(sequence_name)
            seen_in_split.add(sequence_name)
    return normalized


def split_mapping_to_sequences(split_by_sequence: Mapping[str, str]) -> Dict[str, list]:
    splits = make_empty_split_buckets()
    for sequence_name, split in sorted(split_by_sequence.items()):
        if split not in SPLIT_NAMES:
            raise ValueError(f"Invalid split name for {sequence_name}: {split}")
        splits[split].append(sequence_name)
    return splits


def sequence_to_split_roles(split_sequences: Mapping[str, Sequence[str]]) -> Dict[str, list]:
    roles: Dict[str, list] = {}
    for split in SPLIT_NAMES:
        for sequence_name in split_sequences.get(split, []):
            roles.setdefault(sequence_name, []).append(split)
    return roles


def build_split_manifest(
    *,
    dataset: str,
    sequence_paths: Mapping[str, str],
    frame_counts: Mapping[str, int],
    camera_names: Sequence[str],
    max_pose_time_diff_ns: int,
    split_sequences: Optional[Mapping[str, Sequence[str]]] = None,
    split_by_sequence: Optional[Mapping[str, str]] = None,
) -> Dict:
    if split_sequences is None and split_by_sequence is not None:
        normalized_splits = split_mapping_to_sequences(split_by_sequence)
    else:
        normalized_splits = normalize_split_sequences(split_sequences)

    splits = make_empty_split_buckets()
    for split in SPLIT_NAMES:
        for sequence_name in normalized_splits.get(split, []):
            if sequence_name in sequence_paths:
                splits[split].append(sequence_name)

    sequence_to_splits = sequence_to_split_roles(splits)
    sequence_to_split = {
        sequence_name: roles[0]
        for sequence_name, roles in sorted(sequence_to_splits.items())
    }

    sequence_paths_for_manifest = {}
    for split in SPLIT_NAMES:
        for sequence_name in splits[split]:
            sequence_paths_for_manifest[sequence_name] = sequence_paths[sequence_name]

    counts = {}
    for split, sequence_names in splits.items():
        counts[split] = {
            "sequence_count": len(sequence_names),
            "frame_count": int(sum(frame_counts.get(name, 0) for name in sequence_names)),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "split_policy": "sequence",
        "camera_names": list(camera_names),
        "max_pose_time_diff_ns": int(max_pose_time_diff_ns),
        "splits": splits,
        "sequence_to_split": sequence_to_split,
        "sequence_to_splits": sequence_to_splits,
        "sequence_paths": sequence_paths_for_manifest,
        "counts": counts,
    }
    validate_split_manifest(manifest)
    return manifest


def build_sequence_manifest(
    *,
    dataset: str,
    sequence_records: Mapping[str, Mapping],
    camera_names: Sequence[str],
    max_pose_time_diff_ns: int,
) -> Dict:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "split_policy": "configured_in_training",
        "camera_names": list(camera_names),
        "max_pose_time_diff_ns": int(max_pose_time_diff_ns),
        "sequences": dict(sequence_records),
    }
    validate_sequence_manifest(manifest)
    return manifest


def validate_sequence_manifest(manifest: Mapping) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported sequence manifest schema_version: {manifest.get('schema_version')}"
        )
    if manifest.get("split_policy") != "configured_in_training":
        raise ValueError("sequence manifest split_policy must be configured_in_training")
    sequences = manifest.get("sequences")
    if not isinstance(sequences, Mapping):
        raise ValueError("sequence manifest must contain a 'sequences' mapping")
    for sequence_name, record in sequences.items():
        for key in ["file", "sequence_path", "frame_count", "camera_names"]:
            if key not in record:
                raise ValueError(f"sequence manifest record {sequence_name} missing {key}")


def validate_split_manifest(manifest: Mapping) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported split manifest schema_version: {manifest.get('schema_version')}"
        )
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("split manifest must contain a 'splits' mapping")

    sequence_to_splits = {}
    for split in SPLIT_NAMES:
        seen_in_split = set()
        for sequence_name in splits.get(split, []):
            if sequence_name in seen_in_split:
                raise ValueError(f"Sequence {sequence_name} appears twice in {split}")
            seen_in_split.add(sequence_name)
            sequence_to_splits.setdefault(sequence_name, []).append(split)

    manifest_sequence_to_splits = manifest.get("sequence_to_splits", {})
    for sequence_name, splits_for_sequence in manifest_sequence_to_splits.items():
        if sequence_to_splits.get(sequence_name) != splits_for_sequence:
            raise ValueError(
                "sequence_to_splits mismatch for "
                f"{sequence_name}: {splits_for_sequence} != {sequence_to_splits.get(sequence_name)}"
            )


def attach_sequence_metadata(
    *,
    payload: MutableMapping,
    dataset: str,
    sequence_name: str,
    sequence_path: str,
    camera_name: str,
    split: Optional[str] = None,
) -> MutableMapping:
    payload["schema_version"] = SCHEMA_VERSION
    payload["dataset"] = dataset
    payload["sequence_name"] = sequence_name
    payload["sequence_path"] = sequence_path
    payload["camera_name"] = camera_name
    if split is not None:
        payload["split"] = split
    return payload


def add_clean_degradation_defaults(frames: Iterable[MutableMapping]) -> None:
    for frame_id, frame in enumerate(frames):
        frame.setdefault("frame_id", int(frame_id))
        frame.setdefault("clean_image_rel_path", frame["image_rel_path"])
        frame.setdefault(
            "degradation",
            {
                "setting": "clean",
                "variant_id": "clean",
                "metadata_rel_path": None,
            },
        )


def ensure_frame_extrinsics_aliases(frames: Iterable[MutableMapping]) -> None:
    for frame in frames:
        if "extrinsics" not in frame and "extrinsics_w2c" in frame:
            frame["extrinsics"] = deepcopy(frame["extrinsics_w2c"])
        if "extrinsics_w2c" not in frame and "extrinsics" in frame:
            frame["extrinsics_w2c"] = deepcopy(frame["extrinsics"])


def validate_vi_sequence(sequence: Mapping) -> None:
    required_top = [
        "schema_version",
        "dataset",
        "sequence_name",
        "camera_name",
        "sensor",
        "frames",
        "imu_data",
    ]
    for key in required_top:
        if key not in sequence:
            raise ValueError(f"VI sequence missing required key: {key}")

    if sequence["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported VI schema_version: {sequence['schema_version']}")
    if "split" in sequence and sequence["split"] not in SPLIT_NAMES:
        raise ValueError(f"Invalid VI split: {sequence['split']}")

    sensor = sequence["sensor"]
    for key in ["intrinsics", "undistorted_intrinsics", "image_size"]:
        if key not in sensor:
            raise ValueError(f"VI sensor missing required key: {key}")

    frames = sequence["frames"]
    if not isinstance(frames, list):
        raise ValueError("VI sequence frames must be a list")
    for frame in frames:
        for key in ["timestamp_ns", "image_rel_path", "extrinsics"]:
            if key not in frame:
                raise ValueError(f"VI frame missing required key: {key}")

    imu_data = sequence["imu_data"]
    if imu_data is None:
        return
    lengths = {
        key: len(imu_data.get(key, []))
        for key in ["timestamps_ns", "gyro", "accel"]
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"IMU arrays must have equal length, got {lengths}")


def validate_vi_annotation(annotation: Mapping) -> None:
    for sequence in annotation.values():
        validate_vi_sequence(sequence)
