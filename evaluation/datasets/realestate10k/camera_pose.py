import random
import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import torch

from evaluation.common.model import load_model, resolve_inference_dtype
from evaluation.common.metrics import calculate_auc_np, se3_to_relative_pose_error
from vggt.utils.load_fn import load_and_preprocess_images_from_objects
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from training.data.preprocess.generate_local_realestate10k_frames import frame_tolerance_us


def add_arguments(parser):
    parser.add_argument("--split", type=str, default="test", choices=("train", "test"), help="Metadata split to evaluate.")
    parser.add_argument("--fast_eval", action="store_true", default=False, help="Evaluate at most 10 sequences.")
    parser.add_argument("--max_sequences", type=int, default=None, help="Maximum number of valid sequences to evaluate.")
    parser.add_argument("--num_frames", type=int, default=10, help="Frames to sample per sequence.")
    parser.add_argument("--min_num_images", type=int, default=24, help="Minimum local frames required for a sequence.")
    parser.add_argument("--realestate10k_dir", type=str, required=True, help="Path to the RealEstate10K dataset root.")
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
        "--frame_manifest_path",
        type=str,
        default=None,
        help="Optional JSONL manifest produced by generate_local_realestate10k_frames.py.",
    )
    parser.add_argument(
        "--require_frame_manifest",
        action="store_true",
        default=False,
        help="Require a valid frame manifest row for every evaluated RealEstate10K frame.",
    )
    parser.add_argument(
        "--sequence_list_path",
        type=str,
        default=None,
        help=(
            "Optional newline-delimited list of RealEstate10K metadata stems to evaluate, "
            "for example PoseDiffusion's re10k_test_1800.txt subset."
        ),
    )
    parser.add_argument(
        "--gpu_ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional CUDA GPU ids for multi-process RealEstate10K evaluation, e.g. --gpu_ids 4 5 6 7.",
    )


def _extract_youtube_id(url):
    parsed = urlparse(url.strip())
    query_video_ids = parse_qs(parsed.query).get("v")
    if query_video_ids:
        return query_video_ids[0]

    path_parts = [part for part in parsed.path.split("/") if part]
    if path_parts:
        return path_parts[-1]
    return None


def _parse_frame_line(line):
    values = line.strip().split()
    if len(values) != 19:
        return None

    timestamp = values[0]
    intrinsics_normalized = np.asarray([float(value) for value in values[1:5]], dtype=np.float32)
    # RealEstate10K defines P = K [R|t], so the 3x4 block already maps world to camera.
    extrinsics = np.asarray([float(value) for value in values[7:19]], dtype=np.float64).reshape(3, 4)
    return {
        "timestamp": timestamp,
        "intrinsics_normalized": intrinsics_normalized,
        "extrinsics": extrinsics,
    }


def load_frame_manifest(manifest_path):
    manifest = {}
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            video_id = row.get("video_id")
            timestamp = row.get("timestamp")
            if not video_id or timestamp is None:
                raise ValueError(f"Invalid manifest row {line_number} in {manifest_path}: missing video_id or timestamp.")

            row["timestamp"] = str(timestamp)
            manifest[(video_id, str(timestamp))] = row
    return manifest


def load_sequence_list(sequence_list_path):
    sequence_list_path = Path(sequence_list_path)
    sequence_names = []
    seen_names = set()
    duplicate_count = 0

    with sequence_list_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            sequence_name = line.strip()
            if not sequence_name:
                continue

            sequence_names.append(sequence_name)
            if sequence_name in seen_names:
                duplicate_count += 1
            else:
                seen_names.add(sequence_name)

    return {
        "path": str(sequence_list_path),
        "names": tuple(sequence_names),
        "name_set": frozenset(seen_names),
        "requested": len(sequence_names),
        "unique": len(seen_names),
        "duplicates": duplicate_count,
    }


def _resolve_frame_manifest(realestate10k_dir, frame_manifest_path=None, require_frame_manifest=False):
    if frame_manifest_path:
        manifest_path = Path(frame_manifest_path)
    else:
        manifest_path = Path(realestate10k_dir) / "transcode_manifest.jsonl"
        if not manifest_path.is_file():
            if require_frame_manifest:
                raise FileNotFoundError(
                    "RealEstate10K frame manifest is required but was not found: "
                    f"{manifest_path}. Regenerate frames with "
                    "training/data/preprocess/generate_local_realestate10k_frames.py."
                )
            return None, None

    if not manifest_path.is_file():
        if require_frame_manifest:
            raise FileNotFoundError(f"RealEstate10K frame manifest not found: {manifest_path}")
        return None, None
    return load_frame_manifest(manifest_path), str(manifest_path)


def _manifest_row_is_valid(manifest_row):
    try:
        fps = float(manifest_row["fps"])
        abs_error_us = float(manifest_row["abs_error_us"])
        tolerance_us = frame_tolerance_us(fps)
    except (KeyError, TypeError, ValueError):
        return False
    return abs_error_us <= tolerance_us


def _load_sequence_entry(metadata_path, realestate10k_dir, min_num_images, frame_manifest=None, require_frame_manifest=False):
    lines = Path(metadata_path).read_text(encoding="utf-8").splitlines()
    if not lines:
        return None

    video_id = _extract_youtube_id(lines[0])
    if not video_id:
        return None

    image_dir = Path(realestate10k_dir) / "transcode" / video_id
    if not image_dir.is_dir():
        return None

    frames = []
    frame_filter_stats = {
        "missing_image": 0,
        "missing_manifest": 0,
        "stale_manifest": 0,
        "kept": 0,
    }
    for line in lines[1:]:
        parsed_frame = _parse_frame_line(line)
        if parsed_frame is None:
            continue

        image_path = image_dir / f"{parsed_frame['timestamp']}.jpg"
        if not image_path.is_file():
            frame_filter_stats["missing_image"] += 1
            continue

        manifest_row = None
        if frame_manifest is not None or require_frame_manifest:
            manifest_row = (frame_manifest or {}).get((video_id, parsed_frame["timestamp"]))
            if manifest_row is None:
                frame_filter_stats["missing_manifest"] += 1
                continue

            if not _manifest_row_is_valid(manifest_row):
                frame_filter_stats["stale_manifest"] += 1
                continue

        if frame_manifest is not None:
            parsed_frame["frame_manifest"] = manifest_row

        if require_frame_manifest and manifest_row is None:
            frame_filter_stats["missing_manifest"] += 1
            continue

        parsed_frame["image_path"] = str(image_path)
        frames.append(parsed_frame)
        frame_filter_stats["kept"] += 1

    if len(frames) < min_num_images:
        return None

    return {
        "seq_name": Path(metadata_path).stem,
        "metadata_path": str(metadata_path),
        "video_id": video_id,
        "frames": frames,
        "frame_filter_stats": frame_filter_stats,
    }


def load_realestate10k_sequence_entries(
    realestate10k_dir,
    split,
    min_num_images,
    frame_manifest=None,
    require_frame_manifest=False,
    sequence_list=None,
):
    realestate10k_dir = Path(realestate10k_dir)
    split_dir = realestate10k_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"RealEstate10K split directory not found: {split_dir}")

    sequence_entries = []
    metadata_paths = sorted(split_dir.glob("*.txt"))
    if sequence_list is not None:
        requested_names = sequence_list["name_set"]
        metadata_paths = [metadata_path for metadata_path in metadata_paths if metadata_path.stem in requested_names]

    for metadata_path in metadata_paths:
        sequence_entry = _load_sequence_entry(
            metadata_path=metadata_path,
            realestate10k_dir=realestate10k_dir,
            min_num_images=min_num_images,
            frame_manifest=frame_manifest,
            require_frame_manifest=require_frame_manifest,
        )
        if sequence_entry is not None:
            sequence_entries.append(sequence_entry)

    return sequence_entries


def _build_sequence_list_stats(sequence_list, sequence_entries):
    if sequence_list is None:
        return None

    matched_names = {sequence_entry["seq_name"] for sequence_entry in sequence_entries}
    missing_names = sorted(sequence_list["name_set"] - matched_names)
    return {
        "path": sequence_list["path"],
        "requested": sequence_list["requested"],
        "unique": sequence_list["unique"],
        "duplicates": sequence_list["duplicates"],
        "matched": len(matched_names),
        "missing": len(missing_names),
        "missing_names": missing_names,
    }


def load_realestate10k_image_object(image_path):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read RealEstate10K image: {image_path}")
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


def _select_sequence_entries_for_evaluation(
    sequence_entries,
    num_frames,
    fast_eval,
    max_sequences,
    seed,
    min_num_images=None,
):
    random.seed(seed)
    selected_entries = sorted(list(sequence_entries), key=lambda item: item["seq_name"])

    if fast_eval and len(selected_entries) >= 10:
        selected_entries = random.sample(selected_entries, 10)
        selected_entries = sorted(selected_entries, key=lambda item: item["seq_name"])

    if max_sequences is not None:
        selected_entries = selected_entries[:max_sequences]

    filtered_entries = []
    skipped_too_short = 0
    for sequence_entry in selected_entries:
        min_required_frames = min_num_images if min_num_images is not None else num_frames
        if len(sequence_entry["frames"]) < min_required_frames or len(sequence_entry["frames"]) < num_frames:
            skipped_too_short += 1
            continue
        filtered_entries.append(sequence_entry)

    return filtered_entries, skipped_too_short


def _build_sampled_indices_by_seq(sequence_entries, num_frames, seed):
    np.random.seed(seed)
    sampled_indices_by_seq = {}
    for sequence_entry in sequence_entries:
        sampled_indices_by_seq[sequence_entry["seq_name"]] = np.random.choice(
            len(sequence_entry["frames"]), num_frames, replace=False
        )
    return sampled_indices_by_seq


def _shard_sequence_entries(sequence_entries, num_shards):
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}.")

    shards = [[] for _ in range(num_shards)]
    for index, sequence_entry in enumerate(sequence_entries):
        shards[index % num_shards].append(sequence_entry)
    return shards


def _merge_sequence_results(per_sequence, skipped_too_short, frame_filter_stats, thresholds):
    per_sequence = sorted(list(per_sequence), key=lambda item: item["seq_name"])
    empty_results = {
        "num_sequences": 0,
        "per_sequence": per_sequence,
        "skipped_too_short": skipped_too_short,
        "frame_filter_stats": frame_filter_stats,
    }
    for threshold in thresholds:
        threshold_key = int(threshold)
        empty_results[f"RRA@{threshold_key}"] = 0.0
        empty_results[f"RTA@{threshold_key}"] = 0.0
        empty_results[f"AUC@{threshold_key}"] = 0.0

    if not per_sequence:
        return empty_results

    r_errors = np.concatenate([np.asarray(sequence_result["rError"]) for sequence_result in per_sequence])
    t_errors = np.concatenate([np.asarray(sequence_result["tError"]) for sequence_result in per_sequence])
    results = {
        "num_sequences": len(per_sequence),
        "per_sequence": per_sequence,
        "skipped_too_short": skipped_too_short,
        "frame_filter_stats": frame_filter_stats,
    }
    results.update(_metrics_from_errors(r_errors, t_errors, thresholds))
    return results


def _evaluate_sequence_entries_with_samples(
    model,
    sequence_entries,
    sampled_indices_by_seq,
    device,
    dtype,
    thresholds,
    preprocess_mode,
    predictor=None,
):
    per_sequence = []
    for sequence_entry in sequence_entries:
        sequence_result = evaluate_sequence(
            model=model,
            sequence_entry=sequence_entry,
            sampled_indices=sampled_indices_by_seq[sequence_entry["seq_name"]],
            device=device,
            dtype=dtype,
            thresholds=thresholds,
            preprocess_mode=preprocess_mode,
            predictor=predictor,
        )
        per_sequence.append(sequence_result)
    return per_sequence


def _realestate10k_worker(worker_args):
    (
        gpu_id,
        model_path,
        sequence_entries,
        sampled_indices_by_seq,
        thresholds,
        preprocess_mode,
    ) = worker_args
    device = torch.device(f"cuda:{gpu_id}")
    dtype = resolve_inference_dtype(device)
    model = load_model(device, model_path)
    return _evaluate_sequence_entries_with_samples(
        model=model,
        sequence_entries=sequence_entries,
        sampled_indices_by_seq=sampled_indices_by_seq,
        device=device,
        dtype=dtype,
        thresholds=thresholds,
        preprocess_mode=preprocess_mode,
    )


def evaluate_sequences_multi_gpu(
    sequence_entries,
    num_frames,
    fast_eval,
    max_sequences,
    seed,
    gpu_ids,
    model_path,
    thresholds,
    preprocess_mode,
    min_num_images=None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("--gpu_ids requires CUDA, but torch.cuda.is_available() is false.")
    if not gpu_ids:
        raise ValueError("gpu_ids must contain at least one GPU id.")

    selected_entries, skipped_too_short = _select_sequence_entries_for_evaluation(
        sequence_entries=sequence_entries,
        num_frames=num_frames,
        fast_eval=fast_eval,
        max_sequences=max_sequences,
        seed=seed,
        min_num_images=min_num_images,
    )
    sampled_indices_by_seq = _build_sampled_indices_by_seq(selected_entries, num_frames=num_frames, seed=seed)
    shards = _shard_sequence_entries(selected_entries, num_shards=len(gpu_ids))
    worker_args = [
        (
            gpu_id,
            model_path,
            shard,
            {entry["seq_name"]: sampled_indices_by_seq[entry["seq_name"]] for entry in shard},
            thresholds,
            preprocess_mode,
        )
        for gpu_id, shard in zip(gpu_ids, shards)
        if shard
    ]

    per_sequence = []
    if worker_args:
        context = mp.get_context("spawn")
        with context.Pool(processes=len(worker_args)) as pool:
            for shard_results in pool.map(_realestate10k_worker, worker_args):
                per_sequence.extend(shard_results)

    frame_filter_stats = _aggregate_frame_filter_stats(selected_entries)
    return _merge_sequence_results(
        per_sequence=per_sequence,
        skipped_too_short=skipped_too_short,
        frame_filter_stats=frame_filter_stats,
        thresholds=thresholds,
    )


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
    image_objects = [load_realestate10k_image_object(path) for path in image_paths]

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
        "seq_name": sequence_entry["seq_name"],
        "video_id": sequence_entry["video_id"],
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
    sampled_indices_by_seq=None,
    predictor=None,
):
    selected_entries, skipped_too_short = _select_sequence_entries_for_evaluation(
        sequence_entries=sequence_entries,
        num_frames=num_frames,
        fast_eval=fast_eval,
        max_sequences=max_sequences,
        seed=seed,
        min_num_images=min_num_images,
    )
    sampled_indices_by_seq = sampled_indices_by_seq or _build_sampled_indices_by_seq(
        selected_entries, num_frames=num_frames, seed=seed
    )
    per_sequence = _evaluate_sequence_entries_with_samples(
        model=model,
        sequence_entries=selected_entries,
        sampled_indices_by_seq=sampled_indices_by_seq,
        device=device,
        dtype=dtype,
        thresholds=thresholds,
        preprocess_mode=preprocess_mode,
        predictor=predictor,
    )

    frame_filter_stats = _aggregate_frame_filter_stats(selected_entries)
    return _merge_sequence_results(
        per_sequence=per_sequence,
        skipped_too_short=skipped_too_short,
        frame_filter_stats=frame_filter_stats,
        thresholds=thresholds,
    )


def _aggregate_frame_filter_stats(sequence_entries):
    aggregate = {
        "missing_image": 0,
        "missing_manifest": 0,
        "stale_manifest": 0,
        "kept": 0,
    }
    for sequence_entry in sequence_entries:
        for key, value in sequence_entry.get("frame_filter_stats", {}).items():
            aggregate[key] = aggregate.get(key, 0) + int(value)
    return aggregate


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
    return Path(metrics_output_dir) / f"realestate10k_camera_pose_{timestamp}.txt"


def write_metrics_report(output_path, results, thresholds, run_config):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "RealEstate10K camera pose evaluation",
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
    if "skipped_too_short" in results:
        lines.append(f"skipped_too_short: {results['skipped_too_short']}")
    if "frame_filter_stats" in results:
        for key in sorted(results["frame_filter_stats"]):
            lines.append(f"frame_filter_{key}: {results['frame_filter_stats'][key]}")
    if "sequence_list_stats" in results and results["sequence_list_stats"] is not None:
        sequence_list_stats = results["sequence_list_stats"]
        lines.append(f"sequence_list_path: {sequence_list_stats['path']}")
        for key in ("requested", "unique", "matched", "missing", "duplicates"):
            lines.append(f"sequence_list_{key}: {sequence_list_stats[key]}")
        missing_names = sequence_list_stats.get("missing_names") or []
        if missing_names:
            preview = ", ".join(missing_names[:20])
            if len(missing_names) > 20:
                preview += f", ... (+{len(missing_names) - 20} more)"
            lines.append(f"sequence_list_missing_names: {preview}")

    for threshold in thresholds:
        threshold_key = int(threshold)
        lines.append(f"RRA@{threshold_key}: {results[f'RRA@{threshold_key}']:.4f}")
        lines.append(f"RTA@{threshold_key}: {results[f'RTA@{threshold_key}']:.4f}")
        lines.append(f"AUC@{threshold_key}: {results[f'AUC@{threshold_key}']:.4f}")

    lines.extend(["", "[Per sequence]"])
    for sequence_result in results["per_sequence"]:
        lines.append(f"{sequence_result['seq_name']} ({sequence_result['video_id']})")
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
        "realestate10k_dir": args.realestate10k_dir,
        "num_frames": args.num_frames,
        "min_num_images": args.min_num_images,
        "fast_eval": args.fast_eval,
        "max_sequences": args.max_sequences,
        "thresholds": " ".join(str(threshold) for threshold in args.thresholds),
        "preprocess_mode": args.preprocess_mode,
        "seed": args.seed,
        "model_path": args.model_path,
        "frame_manifest_path": args.frame_manifest_path,
        "require_frame_manifest": args.require_frame_manifest,
        "sequence_list_path": args.sequence_list_path,
        "gpu_ids": " ".join(str(gpu_id) for gpu_id in args.gpu_ids) if args.gpu_ids else None,
    }


def should_load_model(args):
    return not bool(args.gpu_ids)


def run(args, model, device, dtype):
    frame_manifest, resolved_manifest_path = _resolve_frame_manifest(
        realestate10k_dir=args.realestate10k_dir,
        frame_manifest_path=args.frame_manifest_path,
        require_frame_manifest=args.require_frame_manifest,
    )
    sequence_list = load_sequence_list(args.sequence_list_path) if args.sequence_list_path else None
    sequence_entries = load_realestate10k_sequence_entries(
        realestate10k_dir=args.realestate10k_dir,
        split=args.split,
        min_num_images=1,
        frame_manifest=frame_manifest,
        require_frame_manifest=args.require_frame_manifest,
        sequence_list=sequence_list,
    )

    if not sequence_entries:
        raise RuntimeError(
            "No valid RealEstate10K sequences found. "
            "Check that transcode/<youtube_id>/<timestamp>.jpg exists for the selected split "
            "and that the frame manifest is valid when --require_frame_manifest is set."
        )

    if args.gpu_ids:
        results = evaluate_sequences_multi_gpu(
            sequence_entries=sequence_entries,
            num_frames=args.num_frames,
            fast_eval=args.fast_eval,
            max_sequences=args.max_sequences,
            seed=args.seed,
            gpu_ids=args.gpu_ids,
            model_path=args.model_path,
            thresholds=tuple(args.thresholds),
            preprocess_mode=args.preprocess_mode,
            min_num_images=args.min_num_images,
        )
    else:
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
    results["sequence_list_stats"] = _build_sequence_list_stats(sequence_list, sequence_entries)

    for sequence_result in results["per_sequence"]:
        print(
            f"{sequence_result['seq_name']} ({sequence_result['video_id']}): "
            f"{_format_threshold_metrics(sequence_result, args.thresholds)}"
        )

    print(
        "RealEstate10K camera pose summary: "
        f"{results['num_sequences']} sequences, "
        f"{_format_threshold_metrics(results, args.thresholds)}"
    )
    if resolved_manifest_path is not None:
        print(f"Using frame manifest: {resolved_manifest_path}")
    if results["sequence_list_stats"] is not None:
        stats = results["sequence_list_stats"]
        print(
            "Using RealEstate10K sequence list: "
            f"{stats['path']} ({stats['matched']}/{stats['unique']} matched, "
            f"{stats['missing']} missing, {stats['duplicates']} duplicates)"
        )
    if args.gpu_ids:
        print(f"Using multi-GPU evaluation on CUDA devices: {args.gpu_ids}")
    print(
        "RealEstate10K frame filters: "
        f"{results['frame_filter_stats']}, skipped_too_short={results['skipped_too_short']}"
    )

    metrics_output_path = _resolve_metrics_output_path(
        metrics_output_path=args.metrics_output_path,
        metrics_output_dir=args.metrics_output_dir,
    )
    write_metrics_report(
        output_path=metrics_output_path,
        results=results,
        thresholds=tuple(args.thresholds),
        run_config={**_build_run_config(args), "resolved_frame_manifest_path": resolved_manifest_path},
    )
    print(f"Metrics report written to: {metrics_output_path}")
    return results
