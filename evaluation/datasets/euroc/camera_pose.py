from datetime import datetime
import os
from pathlib import Path
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
        "--metrics_output_dir",
        type=str,
        default="evaluation/results",
        help="Directory for EuRoC camera pose metric reports.",
    )
    parser.add_argument(
        "--metrics_report_prefix",
        type=str,
        default=None,
        help="Optional metric report filename prefix. Defaults to baseline or IMU-FiLM based on --use_imu.",
    )
    parser.add_argument(
        "--use_imu",
        action="store_true",
        default=False,
        help="Build IMU windows from EuRoC annotations and pass them to VGGT.",
    )
    parser.add_argument(
        "--imu_window_ns",
        type=int,
        default=100_000_000,
        help="Half-width of the IMU sampling window around each frame timestamp.",
    )
    parser.add_argument(
        "--imu_num_samples",
        type=int,
        default=32,
        help="Number of uniformly sampled IMU entries per frame.",
    )
    parser.add_argument(
        "--imu_feature_order",
        type=str,
        choices=("gyro_accel", "accel_gyro"),
        default="gyro_accel",
        help="Feature order for IMU windows passed to the model.",
    )
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
        "distortion_model": str(sensor.get("distortion_model", "radial-tangential")),
    }


def _deserialize_imu_data(imu_data):
    if imu_data is None:
        return None
    return {
        "timestamps_ns": np.asarray(imu_data["timestamps_ns"], dtype=np.int64),
        "gyro": np.asarray(imu_data["gyro"], dtype=np.float32),
        "accel": np.asarray(imu_data["accel"], dtype=np.float32),
    }


def _build_euroc_sequence_entries(raw_annotation, euroc_dir, camera_names, min_num_images):
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
                "imu_data": _deserialize_imu_data(payload.get("imu_data")),
                "frames": frames,
            }
        )

    return sequence_entries


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
    return _build_euroc_sequence_entries(
        raw_annotation=raw_annotation,
        euroc_dir=euroc_dir,
        camera_names=camera_names,
        min_num_images=min_num_images,
    )


def load_euroc_image_object(image_path, sensor, undistort_images):
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read EuRoC image: {image_path}")

    if undistort_images:
        if sensor.get("distortion_model") == "equidistant":
            image_bgr = cv2.fisheye.undistortImage(
                image_bgr,
                sensor["intrinsics"],
                sensor["distortion"].reshape(-1, 1),
                Knew=sensor["undistorted_intrinsics"],
            )
        else:
            image_bgr = cv2.undistort(
                image_bgr,
                sensor["intrinsics"],
                sensor["distortion"],
                None,
                sensor["undistorted_intrinsics"],
            )

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _empty_imu_window(imu_num_samples):
    return (
        np.zeros((imu_num_samples, 6), dtype=np.float32),
        np.zeros((imu_num_samples,), dtype=bool),
    )


def _format_imu_features(gyro, accel, imu_feature_order):
    if imu_feature_order == "gyro_accel":
        return np.concatenate([gyro, accel], axis=-1).astype(np.float32)
    if imu_feature_order == "accel_gyro":
        return np.concatenate([accel, gyro], axis=-1).astype(np.float32)
    raise ValueError(f"Unsupported imu_feature_order: {imu_feature_order}")


def build_imu_window(
    imu_data,
    center_timestamp_ns,
    imu_window_ns=100_000_000,
    imu_num_samples=32,
    imu_feature_order="gyro_accel",
):
    if imu_num_samples <= 0:
        raise ValueError(f"imu_num_samples must be positive, got {imu_num_samples}")
    if imu_data is None:
        return _empty_imu_window(imu_num_samples)

    timestamps = imu_data["timestamps_ns"]
    if len(timestamps) == 0:
        return _empty_imu_window(imu_num_samples)

    start_ts = int(center_timestamp_ns) - int(imu_window_ns)
    end_ts = int(center_timestamp_ns) + int(imu_window_ns)
    left = int(np.searchsorted(timestamps, start_ts, side="left"))
    right = int(np.searchsorted(timestamps, end_ts, side="right"))
    window_timestamps = timestamps[left:right]
    if len(window_timestamps) == 0:
        return _empty_imu_window(imu_num_samples)

    target_timestamps = np.linspace(
        start_ts,
        end_ts,
        num=imu_num_samples,
        dtype=np.float64,
    )
    gyro = np.zeros((imu_num_samples, 3), dtype=np.float32)
    accel = np.zeros((imu_num_samples, 3), dtype=np.float32)
    valid_mask = (
        (target_timestamps >= float(window_timestamps[0]))
        & (target_timestamps <= float(window_timestamps[-1]))
    )

    # 预计算 float64 时间轴，避免在 6 次 np.interp 调用中重复转换
    window_ts_f64 = window_timestamps.astype(np.float64)
    imu_gyro = imu_data["gyro"][left:right]
    imu_accel = imu_data["accel"][left:right]
    for axis in range(3):
        gyro[:, axis] = np.interp(
            target_timestamps,
            window_ts_f64,
            imu_gyro[:, axis].astype(np.float64),
        ).astype(np.float32)
        accel[:, axis] = np.interp(
            target_timestamps,
            window_ts_f64,
            imu_accel[:, axis].astype(np.float64),
        ).astype(np.float32)

    imu_window = _format_imu_features(gyro, accel, imu_feature_order)
    imu_window[~valid_mask] = 0.0
    return imu_window, valid_mask.astype(bool)


def attach_imu_windows_to_frames(
    frame_entries,
    imu_data,
    imu_window_ns=100_000_000,
    imu_num_samples=32,
    imu_feature_order="gyro_accel",
):
    updated_entries = []
    for frame in frame_entries:
        frame = dict(frame)
        imu_window, imu_window_mask = build_imu_window(
            imu_data=imu_data,
            center_timestamp_ns=frame["timestamp_ns"],
            imu_window_ns=imu_window_ns,
            imu_num_samples=imu_num_samples,
            imu_feature_order=imu_feature_order,
        )
        frame["imu_window"] = imu_window
        frame["imu_window_mask"] = imu_window_mask
        updated_entries.append(frame)
    return updated_entries


def predict_camera_extrinsics(
    model,
    image_paths,
    image_objects,
    frame_entries,
    device,
    dtype,
):
    del image_paths
    images = load_and_preprocess_images_from_objects(image_objects).to(device)
    model_kwargs = {"images": images}
    if frame_entries and "imu_window" in frame_entries[0]:
        model_kwargs["imu_windows"] = torch.from_numpy(
            np.stack([frame["imu_window"] for frame in frame_entries], axis=0)
        ).to(device=device, dtype=images.dtype)
        model_kwargs["imu_window_masks"] = torch.from_numpy(
            np.stack([frame["imu_window_mask"] for frame in frame_entries], axis=0)
        ).to(device=device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(**model_kwargs)
        else:
            predictions = model(**model_kwargs)

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
    use_imu=False,
    imu_window_ns=100_000_000,
    imu_num_samples=32,
    imu_feature_order="gyro_accel",
):
    frame_entries = [sequence_entry["frames"][index] for index in sampled_indices]
    if use_imu:
        frame_entries = attach_imu_windows_to_frames(
            frame_entries=frame_entries,
            imu_data=sequence_entry.get("imu_data"),
            imu_window_ns=imu_window_ns,
            imu_num_samples=imu_num_samples,
            imu_feature_order=imu_feature_order,
        )
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
    use_imu=False,
    imu_window_ns=100_000_000,
    imu_num_samples=32,
    imu_feature_order="gyro_accel",
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
            use_imu=use_imu,
            imu_window_ns=imu_window_ns,
            imu_num_samples=imu_num_samples,
            imu_feature_order=imu_feature_order,
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


def _evaluate_sequence_entries(args, model, device, dtype, sequence_entries, predictor=None):
    return evaluate_sequences(
        model=model,
        sequence_entries=sequence_entries,
        num_frames=args.num_frames,
        fast_eval=args.fast_eval,
        seed=args.seed,
        device=device,
        dtype=dtype,
        undistort_images=not args.no_undistort,
        predictor=predictor,
        use_imu=getattr(args, "use_imu", False),
        imu_window_ns=getattr(args, "imu_window_ns", 100_000_000),
        imu_num_samples=getattr(args, "imu_num_samples", 32),
        imu_feature_order=getattr(args, "imu_feature_order", "gyro_accel"),
    )


def evaluate_euroc_variants(args, model, device, dtype, predictor=None):
    annotation_path = os.path.join(args.euroc_anno_dir, f"euroc_{args.split}.jgz")
    raw_annotation = load_json_gz(annotation_path)

    clean_entries = _build_euroc_sequence_entries(
        raw_annotation=raw_annotation,
        euroc_dir=args.euroc_dir,
        camera_names=tuple(args.camera_names),
        min_num_images=args.min_num_images,
    )
    variant_results = {
        "clean": _evaluate_sequence_entries(
            args=args,
            model=model,
            device=device,
            dtype=dtype,
            sequence_entries=clean_entries,
            predictor=predictor,
        )
    }

    return variant_results


def _format_variant_lines(variant_name, results):
    lines = [f"EuRoC variant: {variant_name}"]
    for sequence_result in results["per_sequence"]:
        lines.append(
            f"{sequence_result['seq_name']} R_ACC@5: {sequence_result['R_ACC@5']:.4f} "
            f"T_ACC@5: {sequence_result['T_ACC@5']:.4f}"
        )
    lines.append(
        "EuRoC camera pose summary: "
        f"{results['AUC@30']:.4f} (AUC@30), "
        f"{results['AUC@15']:.4f} (AUC@15), "
        f"{results['AUC@5']:.4f} (AUC@5), "
        f"{results['AUC@3']:.4f} (AUC@3)"
    )
    return lines


def _format_summary_lines(variant_results):
    lines = ["EuRoC camera pose variant summary:"]
    for variant_name, results in variant_results.items():
        lines.append(
            f"{variant_name}: "
            f"{results['AUC@30']:.4f} (AUC@30), "
            f"{results['AUC@15']:.4f} (AUC@15), "
            f"{results['AUC@5']:.4f} (AUC@5), "
            f"{results['AUC@3']:.4f} (AUC@3)"
        )
    return lines


def write_euroc_metrics_report(args, variant_results):
    output_dir = Path(args.metrics_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_prefix = getattr(args, "metrics_report_prefix", None)
    if report_prefix is None:
        report_prefix = "euroc_imu_film_test" if getattr(args, "use_imu", False) else "euroc_baseline_test"
    output_path = output_dir / f"{report_prefix}_{timestamp}.txt"
    suffix = 1
    while output_path.exists():
        output_path = output_dir / f"{report_prefix}_{timestamp}_{suffix}.txt"
        suffix += 1

    lines = [
        "EuRoC camera pose IMU-FiLM evaluation" if getattr(args, "use_imu", False) else "EuRoC camera pose baseline evaluation",
        f"split: {args.split}",
        f"euroc_dir: {args.euroc_dir}",
        f"euroc_anno_dir: {args.euroc_anno_dir}",
        f"use_imu: {getattr(args, 'use_imu', False)}",
        f"imu_window_ns: {getattr(args, 'imu_window_ns', None)}",
        f"imu_num_samples: {getattr(args, 'imu_num_samples', None)}",
        f"imu_feature_order: {getattr(args, 'imu_feature_order', None)}",
        f"num_frames: {args.num_frames}",
        f"seed: {args.seed}",
        "",
    ]
    for variant_name, results in variant_results.items():
        lines.extend(_format_variant_lines(variant_name, results))
        lines.append("")
    lines.extend(_format_summary_lines(variant_results))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run(args, model, device, dtype):
    variant_results = evaluate_euroc_variants(
        args=args,
        model=model,
        device=device,
        dtype=dtype,
    )

    for variant_name, results in variant_results.items():
        for line in _format_variant_lines(variant_name, results):
            print(line)
        print("")

    for line in _format_summary_lines(variant_results):
        print(line)

    report_path = write_euroc_metrics_report(args, variant_results)
    print(f"Saved EuRoC metrics report to: {report_path}")
    if len(variant_results) == 1:
        return next(iter(variant_results.values()))
    return variant_results
