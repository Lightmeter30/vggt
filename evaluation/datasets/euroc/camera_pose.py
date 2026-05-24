import os
import random

import cv2
import numpy as np
import torch

from evaluation.common.io import load_json_gz
from evaluation.common.metrics import calculate_auc_np, se3_to_relative_pose_error
from vggt.utils.load_fn import load_and_preprocess_images_from_objects
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def add_arguments(parser):
    parser.add_argument("--split", type=str, default="test", help="Annotation split to evaluate.")
    parser.add_argument("--fast_eval", action="store_true", default=False, help="Evaluate at most 10 sequences.")
    parser.add_argument("--num_frames", type=int, default=10, help="Frames to sample per sequence.")
    parser.add_argument("--min_num_images", type=int, default=24, help="Minimum images required for a sequence.")
    parser.add_argument("--euroc_dir", type=str, required=True, help="Path to the EuRoC dataset root.")
    parser.add_argument("--euroc_anno_dir", type=str, required=True, help="Path to EuRoC annotations.")
    parser.add_argument(
        "--camera_names",
        nargs="+",
        default=["cam0"],
        help="Camera names to evaluate, e.g. cam0 cam1.",
    )
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        default=False,
        help="Disable image undistortion before VGGT preprocessing.",
    )


def _deserialize_sensor(sensor):
    return {
        "intrinsics": np.asarray(sensor["intrinsics"], dtype=np.float32),
        "distortion": np.asarray(sensor["distortion"], dtype=np.float32),
        "undistorted_intrinsics": np.asarray(
            sensor["undistorted_intrinsics"], dtype=np.float32
        ),
        "image_size": np.asarray(sensor["image_size"], dtype=np.int32),
    }


def _deserialize_frame(frame, euroc_dir):
    return {
        "timestamp_ns": int(frame["timestamp_ns"]),
        "gt_timestamp_ns": int(frame["gt_timestamp_ns"]),
        "pose_dt_ns": int(frame["pose_dt_ns"]),
        "image_rel_path": frame["image_rel_path"],
        "image_path": os.path.join(str(euroc_dir), frame["image_rel_path"]),
        "extrinsics": np.asarray(
            frame.get("extrinsics", frame.get("extrinsics_w2c")), dtype=np.float64
        ),
    }


def load_euroc_sequence_entries(annotation_path, euroc_dir, camera_names, min_num_images):
    raw_annotation = load_json_gz(annotation_path)
    camera_filter = set(camera_names or [])
    sequence_entries = []

    for seq_name, payload in sorted(raw_annotation.items()):
        if camera_filter and payload["camera_name"] not in camera_filter:
            continue

        frames = [_deserialize_frame(frame, euroc_dir) for frame in payload["frames"]]
        if len(frames) < min_num_images:
            continue

        sequence_entries.append(
            {
                "seq_name": seq_name,
                "camera_name": payload["camera_name"],
                "sensor": _deserialize_sensor(payload["sensor"]),
                "frames": frames,
            }
        )

    return sequence_entries


def load_euroc_image_object(image_path, sensor, undistort_images):
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read EuRoC image: {image_path}")

    if undistort_images:
        image_bgr = cv2.undistort(
            image_bgr,
            sensor["intrinsics"],
            sensor["distortion"],
            None,
            sensor["undistorted_intrinsics"],
        )

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def predict_camera_extrinsics(
    model,
    image_paths,
    image_objects,
    frame_entries,
    device,
    dtype,
):
    del image_paths, frame_entries
    images = load_and_preprocess_images_from_objects(image_objects).to(device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)

    pose_encoding = predictions["pose_enc"].to(torch.float64)
    extrinsic, _ = pose_encoding_to_extri_intri(pose_encoding, images.shape[-2:])
    return extrinsic[0].to(torch.float64)


def evaluate_sequence(
    model,
    sequence_entry,
    sampled_indices,
    device,
    dtype,
    undistort_images,
    predictor=None,
):
    frame_entries = [sequence_entry["frames"][index] for index in sampled_indices]
    image_paths = [frame["image_path"] for frame in frame_entries]
    image_objects = [
        load_euroc_image_object(path, sequence_entry["sensor"], undistort_images)
        for path in image_paths
    ]

    predictor = predictor or predict_camera_extrinsics
    pred_extrinsic = predictor(
        model=model,
        image_paths=image_paths,
        image_objects=image_objects,
        frame_entries=frame_entries,
        device=device,
        dtype=dtype,
    ).to(device=device, dtype=torch.float64)

    gt_extrinsic = torch.tensor(
        np.stack([frame["extrinsics"] for frame in frame_entries], axis=0),
        device=device,
        dtype=torch.float64,
    )
    add_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float64).expand(
        pred_extrinsic.size(0), 1, 4
    )

    pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
    gt_se3 = torch.cat((gt_extrinsic, add_row), dim=1)
    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
        pred_se3, gt_se3, len(frame_entries)
    )

    return {
        "seq_name": sequence_entry["seq_name"],
        "frame_indices": sampled_indices.tolist(),
        "rError": rel_rangle_deg.cpu().numpy(),
        "tError": rel_tangle_deg.cpu().numpy(),
        "R_ACC@5": (rel_rangle_deg < 5).double().mean().item(),
        "T_ACC@5": (rel_tangle_deg < 5).double().mean().item(),
    }


def evaluate_sequences(
    model,
    sequence_entries,
    num_frames,
    fast_eval,
    seed,
    device,
    dtype,
    undistort_images,
    predictor=None,
):
    python_rng = random.Random(seed)
    numpy_rng = np.random.default_rng(seed)
    selected_entries = list(sequence_entries)

    if fast_eval and len(selected_entries) > 10:
        selected_entries = python_rng.sample(selected_entries, 10)
        selected_entries = sorted(selected_entries, key=lambda item: item["seq_name"])

    per_sequence = []
    r_errors = []
    t_errors = []

    for sequence_entry in selected_entries:
        if len(sequence_entry["frames"]) < num_frames:
            continue

        sampled_indices = np.sort(
            numpy_rng.choice(len(sequence_entry["frames"]), size=num_frames, replace=False)
        )
        sequence_result = evaluate_sequence(
            model=model,
            sequence_entry=sequence_entry,
            sampled_indices=sampled_indices,
            device=device,
            dtype=dtype,
            undistort_images=undistort_images,
            predictor=predictor,
        )
        per_sequence.append(sequence_result)
        r_errors.extend(sequence_result["rError"])
        t_errors.extend(sequence_result["tError"])

    if not r_errors:
        return {
            "num_sequences": 0,
            "per_sequence": per_sequence,
            "AUC@30": 0.0,
            "AUC@15": 0.0,
            "AUC@5": 0.0,
            "AUC@3": 0.0,
        }

    r_errors = np.asarray(r_errors)
    t_errors = np.asarray(t_errors)
    auc_30, _ = calculate_auc_np(r_errors, t_errors, max_threshold=30)
    auc_15, _ = calculate_auc_np(r_errors, t_errors, max_threshold=15)
    auc_5, _ = calculate_auc_np(r_errors, t_errors, max_threshold=5)
    auc_3, _ = calculate_auc_np(r_errors, t_errors, max_threshold=3)

    return {
        "num_sequences": len(per_sequence),
        "per_sequence": per_sequence,
        "AUC@30": auc_30,
        "AUC@15": auc_15,
        "AUC@5": auc_5,
        "AUC@3": auc_3,
    }


def run(args, model, device, dtype):
    annotation_path = os.path.join(args.euroc_anno_dir, f"euroc_{args.split}.jgz")
    sequence_entries = load_euroc_sequence_entries(
        annotation_path=annotation_path,
        euroc_dir=args.euroc_dir,
        camera_names=tuple(args.camera_names),
        min_num_images=args.min_num_images,
    )

    results = evaluate_sequences(
        model=model,
        sequence_entries=sequence_entries,
        num_frames=args.num_frames,
        fast_eval=args.fast_eval,
        seed=args.seed,
        device=device,
        dtype=dtype,
        undistort_images=not args.no_undistort,
    )

    for sequence_result in results["per_sequence"]:
        print(
            f"{sequence_result['seq_name']} R_ACC@5: {sequence_result['R_ACC@5']:.4f} "
            f"T_ACC@5: {sequence_result['T_ACC@5']:.4f}"
        )

    print(
        "EuRoC camera pose summary: "
        f"{results['AUC@30']:.4f} (AUC@30), "
        f"{results['AUC@15']:.4f} (AUC@15), "
        f"{results['AUC@5']:.4f} (AUC@5), "
        f"{results['AUC@3']:.4f} (AUC@3)"
    )
    return results
