import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluation.common.io import load_json_gz
from evaluation.common.metrics import calculate_auc_np, se3_to_relative_pose_error
from vggt.utils.load_fn import load_and_preprocess_images_from_objects
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


CO3D_SEEN_CATEGORIES = (
    "apple",
    "backpack",
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "cup",
    "donut",
    "hairdryer",
    "handbag",
    "hydrant",
    "keyboard",
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "stopsign",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
)


def add_arguments(parser):
    parser.add_argument("--split", type=str, default="test", choices=("train", "test"), help="Annotation split to evaluate.")
    parser.add_argument("--fast_eval", action="store_true", default=False, help="Evaluate at most 10 sequences.")
    parser.add_argument("--max_sequences", type=int, default=None, help="Maximum number of valid sequences to evaluate.")
    parser.add_argument("--num_frames", type=int, default=10, help="Frames to sample per sequence.")
    parser.add_argument("--min_num_images", type=int, default=10, help="Minimum local frames required for a sequence.")
    parser.add_argument("--co3d_dir", type=str, required=True, help="Path to the local Co3D dataset root.")
    parser.add_argument("--co3d_anno_dir", type=str, required=True, help="Path to local Co3D annotations.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["all"],
        help="Co3D categories to evaluate, or 'all' to scan annotation files.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=[3, 5, 15, 30],
        help="Angle thresholds for RRA@N, RTA@N, and AUC@N.",
    )
    parser.add_argument(
        "--preprocess_mode",
        type=str,
        default="crop",
        choices=("crop", "pad"),
        help="VGGT image preprocessing mode.",
    )
    parser.add_argument(
        "--metrics_output_path",
        type=str,
        default=None,
        help="Optional text file path for saving evaluation metrics.",
    )
    parser.add_argument(
        "--metrics_output_dir",
        type=str,
        default="evaluation/results",
        help="Directory for auto-generated metrics text files when --metrics_output_path is not set.",
    )


def convert_pt3d_rt_to_opencv(rotation, translation):
    rot_pt3d = np.asarray(rotation, dtype=np.float64)
    trans_pt3d = np.asarray(translation, dtype=np.float64).copy()
    trans_pt3d[:2] *= -1
    rot_pt3d[:, :2] *= -1
    return np.hstack((rot_pt3d.transpose(1, 0), trans_pt3d[:, None]))


def _resolve_categories(co3d_anno_dir, split, categories):
    requested_categories = tuple(categories or ("all",))
    if "all" not in requested_categories:
        return requested_categories

    anno_dir = Path(co3d_anno_dir)
    discovered = sorted(path.name[: -len(f"_{split}.jgz")] for path in anno_dir.glob(f"*_{split}.jgz"))
    if discovered:
        return tuple(discovered)
    return CO3D_SEEN_CATEGORIES


def _deserialize_frame(frame, co3d_dir):
    if "extri" in frame:
        extrinsics = np.asarray(frame["extri"], dtype=np.float64)
    elif "R" in frame and "T" in frame:
        if sum(frame["T"]) > 1e5:
            return None
        extrinsics = convert_pt3d_rt_to_opencv(frame["R"], frame["T"])
    else:
        raise ValueError("Co3D frame annotation must contain 'extri' or legacy 'R'/'T'.")

    image_path = Path(co3d_dir) / frame["filepath"]
    if not image_path.is_file():
        return None

    return {
        "image_rel_path": frame["filepath"],
        "image_path": str(image_path),
        "extrinsics": extrinsics,
        "intrinsics": np.asarray(frame.get("intri"), dtype=np.float64) if "intri" in frame else None,
    }


def load_co3d_sequence_entries(co3d_dir, co3d_anno_dir, split, categories=("all",), min_num_images=10):
    sequence_entries = []
    co3d_anno_dir = Path(co3d_anno_dir)

    for category in _resolve_categories(co3d_anno_dir, split, categories):
        annotation_path = co3d_anno_dir / f"{category}_{split}.jgz"
        if not annotation_path.is_file():
            continue

        annotation = load_json_gz(annotation_path)
        for seq_name, seq_frames in sorted(annotation.items()):
            frames = []
            for frame in seq_frames:
                deserialized = _deserialize_frame(frame, co3d_dir)
                if deserialized is not None:
                    frames.append(deserialized)

            if len(frames) < min_num_images:
                continue

            sequence_entries.append(
                {
                    "category": category,
                    "seq_name": seq_name,
                    "frames": frames,
                }
            )

    return sorted(sequence_entries, key=lambda item: (item["category"], item["seq_name"]))


def load_co3d_image_object(image_path):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read Co3D image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def predict_camera_extrinsics(
    model,
    image_paths,
    image_objects,
    frame_entries,
    device,
    dtype,
    preprocess_mode="crop",
):
    del image_paths, frame_entries
    device = torch.device(device)
    images = load_and_preprocess_images_from_objects(image_objects, mode=preprocess_mode).to(device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)

    pose_encoding = predictions["pose_enc"].to(torch.float64)
    extrinsic, _ = pose_encoding_to_extri_intri(pose_encoding, images.shape[-2:])
    return extrinsic[0].to(torch.float64)


def _metrics_from_errors(r_errors, t_errors, thresholds):
    results = {}
    for threshold in thresholds:
        threshold_key = int(threshold)
        results[f"RRA@{threshold_key}"] = float((r_errors < threshold_key).mean())
        results[f"RTA@{threshold_key}"] = float((t_errors < threshold_key).mean())
        auc, _ = calculate_auc_np(r_errors, t_errors, max_threshold=threshold_key)
        results[f"AUC@{threshold_key}"] = float(auc)
    return results


def evaluate_sequence(
    model,
    sequence_entry,
    sampled_indices,
    device,
    dtype,
    thresholds,
    preprocess_mode,
    predictor=None,
):
    frame_entries = [sequence_entry["frames"][index] for index in sampled_indices]
    image_paths = [frame["image_path"] for frame in frame_entries]
    image_objects = [load_co3d_image_object(path) for path in image_paths]

    predictor = predictor or predict_camera_extrinsics
    pred_extrinsic = predictor(
        model=model,
        image_paths=image_paths,
        image_objects=image_objects,
        frame_entries=frame_entries,
        device=device,
        dtype=dtype,
        preprocess_mode=preprocess_mode,
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

    r_errors = rel_rangle_deg.cpu().numpy()
    t_errors = rel_tangle_deg.cpu().numpy()
    sequence_result = {
        "category": sequence_entry["category"],
        "seq_name": sequence_entry["seq_name"],
        "frame_indices": sampled_indices.tolist(),
        "rError": r_errors,
        "tError": t_errors,
    }
    sequence_result.update(_metrics_from_errors(r_errors, t_errors, thresholds))
    return sequence_result


def evaluate_sequences(
    model,
    sequence_entries,
    num_frames,
    fast_eval,
    max_sequences,
    seed,
    device,
    dtype,
    thresholds,
    preprocess_mode,
    min_num_images=None,
    predictor=None,
):
    random.seed(seed)
    np.random.seed(seed)
    selected_entries = sorted(list(sequence_entries), key=lambda item: (item["category"], item["seq_name"]))

    if fast_eval and len(selected_entries) >= 10:
        selected_entries = random.sample(selected_entries, 10)
        selected_entries = sorted(selected_entries, key=lambda item: (item["category"], item["seq_name"]))

    if max_sequences is not None:
        selected_entries = selected_entries[:max_sequences]

    per_sequence = []
    r_errors = []
    t_errors = []

    for sequence_entry in selected_entries:
        min_required_frames = min_num_images if min_num_images is not None else num_frames
        if len(sequence_entry["frames"]) < min_required_frames or len(sequence_entry["frames"]) < num_frames:
            continue

        sampled_indices = np.random.choice(len(sequence_entry["frames"]), num_frames, replace=False)
        sequence_result = evaluate_sequence(
            model=model,
            sequence_entry=sequence_entry,
            sampled_indices=sampled_indices,
            device=device,
            dtype=dtype,
            thresholds=thresholds,
            preprocess_mode=preprocess_mode,
            predictor=predictor,
        )
        per_sequence.append(sequence_result)
        r_errors.extend(sequence_result["rError"])
        t_errors.extend(sequence_result["tError"])

    empty_results = {"num_sequences": 0, "per_sequence": per_sequence}
    for threshold in thresholds:
        threshold_key = int(threshold)
        empty_results[f"RRA@{threshold_key}"] = 0.0
        empty_results[f"RTA@{threshold_key}"] = 0.0
        empty_results[f"AUC@{threshold_key}"] = 0.0

    if not r_errors:
        return empty_results

    r_errors = np.asarray(r_errors)
    t_errors = np.asarray(t_errors)
    results = {
        "num_sequences": len(per_sequence),
        "per_sequence": per_sequence,
    }
    results.update(_metrics_from_errors(r_errors, t_errors, thresholds))
    return results


def _format_threshold_metrics(results, thresholds):
    fields = []
    for threshold in thresholds:
        threshold_key = int(threshold)
        fields.append(f"{results[f'RRA@{threshold_key}']:.4f} (RRA@{threshold_key})")
        fields.append(f"{results[f'RTA@{threshold_key}']:.4f} (RTA@{threshold_key})")
        fields.append(f"{results[f'AUC@{threshold_key}']:.4f} (AUC@{threshold_key})")
    return ", ".join(fields)


def _format_error_stats(prefix, errors):
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0:
        return [
            f"mean_{prefix}_error_deg: 0.0000",
            f"median_{prefix}_error_deg: 0.0000",
            f"max_{prefix}_error_deg: 0.0000",
        ]
    return [
        f"mean_{prefix}_error_deg: {float(np.mean(errors)):.4f}",
        f"median_{prefix}_error_deg: {float(np.median(errors)):.4f}",
        f"max_{prefix}_error_deg: {float(np.max(errors)):.4f}",
    ]


def _resolve_metrics_output_path(metrics_output_path, metrics_output_dir):
    if metrics_output_path:
        return Path(metrics_output_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(metrics_output_dir) / f"co3d_camera_pose_{timestamp}.txt"


def write_metrics_report(output_path, results, thresholds, run_config):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Co3D camera pose evaluation",
        "",
        "[Run config]",
    ]
    for key in sorted(run_config):
        lines.append(f"{key}: {run_config[key]}")

    lines.extend(
        [
            "",
            "[Summary]",
            f"num_sequences: {results['num_sequences']}",
        ]
    )
    for threshold in thresholds:
        threshold_key = int(threshold)
        lines.append(f"RRA@{threshold_key}: {results[f'RRA@{threshold_key}']:.4f}")
        lines.append(f"RTA@{threshold_key}: {results[f'RTA@{threshold_key}']:.4f}")
        lines.append(f"AUC@{threshold_key}: {results[f'AUC@{threshold_key}']:.4f}")

    lines.extend(["", "[Per sequence]"])
    for sequence_result in results["per_sequence"]:
        lines.append(f"{sequence_result['category']}/{sequence_result['seq_name']}")
        lines.append(f"frame_indices: {sequence_result['frame_indices']}")
        lines.extend(_format_error_stats("r", sequence_result["rError"]))
        lines.extend(_format_error_stats("t", sequence_result["tError"]))
        for threshold in thresholds:
            threshold_key = int(threshold)
            lines.append(f"RRA@{threshold_key}: {sequence_result[f'RRA@{threshold_key}']:.4f}")
            lines.append(f"RTA@{threshold_key}: {sequence_result[f'RTA@{threshold_key}']:.4f}")
            lines.append(f"AUC@{threshold_key}: {sequence_result[f'AUC@{threshold_key}']:.4f}")
        lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def _build_run_config(args):
    return {
        "split": args.split,
        "co3d_dir": args.co3d_dir,
        "co3d_anno_dir": args.co3d_anno_dir,
        "categories": " ".join(args.categories),
        "num_frames": args.num_frames,
        "min_num_images": args.min_num_images,
        "fast_eval": args.fast_eval,
        "max_sequences": args.max_sequences,
        "thresholds": " ".join(str(threshold) for threshold in args.thresholds),
        "preprocess_mode": args.preprocess_mode,
        "seed": args.seed,
        "model_path": args.model_path,
    }


def run(args, model, device, dtype):
    sequence_entries = load_co3d_sequence_entries(
        co3d_dir=args.co3d_dir,
        co3d_anno_dir=args.co3d_anno_dir,
        split=args.split,
        categories=tuple(args.categories),
        min_num_images=1,
    )

    if not sequence_entries:
        raise RuntimeError(
            "No valid Co3D sequences found. Check --co3d_dir, --co3d_anno_dir, "
            "--split, and --categories. Local annotations must contain frames with "
            "'filepath' and OpenCV 'extri' entries."
        )

    results = evaluate_sequences(
        model=model,
        sequence_entries=sequence_entries,
        num_frames=args.num_frames,
        fast_eval=args.fast_eval,
        max_sequences=args.max_sequences,
        seed=args.seed,
        device=device,
        dtype=dtype,
        thresholds=tuple(args.thresholds),
        preprocess_mode=args.preprocess_mode,
        min_num_images=args.min_num_images,
    )

    for sequence_result in results["per_sequence"]:
        print(
            f"{sequence_result['category']}/{sequence_result['seq_name']}: "
            f"{_format_threshold_metrics(sequence_result, args.thresholds)}"
        )

    print(
        "Co3D camera pose summary: "
        f"{results['num_sequences']} sequences, "
        f"{_format_threshold_metrics(results, args.thresholds)}"
    )

    metrics_output_path = _resolve_metrics_output_path(
        metrics_output_path=args.metrics_output_path,
        metrics_output_dir=args.metrics_output_dir,
    )
    write_metrics_report(
        output_path=metrics_output_path,
        results=results,
        thresholds=tuple(args.thresholds),
        run_config=_build_run_config(args),
    )
    print(f"Metrics report written to: {metrics_output_path}")
    return results
