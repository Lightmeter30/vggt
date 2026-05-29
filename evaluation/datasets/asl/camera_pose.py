import gzip
import json
from datetime import datetime
from pathlib import Path

from evaluation.datasets.euroc.camera_pose import (
    _build_euroc_sequence_entries,
    evaluate_sequences,
)


def add_arguments(parser):
    parser.add_argument("--split", type=str, default="test", help="Annotation split to evaluate.")
    parser.add_argument("--fast_eval", action="store_true", default=False, help="Evaluate at most 10 sequences.")
    parser.add_argument("--num_frames", type=int, default=10, help="Frames to sample per sequence.")
    parser.add_argument("--min_num_images", type=int, default=24, help="Minimum images required for a sequence.")
    parser.add_argument(
        "--metrics_output_dir",
        type=str,
        default="evaluation/results",
        help="Directory for ASL camera pose metric reports.",
    )
    parser.add_argument(
        "--metrics_report_prefix",
        type=str,
        default=None,
        help="Optional metric report filename prefix.",
    )
    parser.add_argument(
        "--use_imu",
        action="store_true",
        default=False,
        help="Build IMU windows from ASL annotations and pass them to VGGT.",
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
        default=None,
        help="Global camera name filter used when a dataset spec does not set camera_names.",
    )
    parser.add_argument(
        "--sequence_names",
        nargs="+",
        default=None,
        help="CLI sequence filter for the single-dataset --asl_dir path.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="asl",
        help="Dataset name for the single-dataset --asl_dir path.",
    )
    parser.add_argument(
        "--asl_dir",
        type=str,
        default=None,
        help="Single ASL dataset root. Prefer YAML datasets for multi-dataset evaluation.",
    )
    parser.add_argument(
        "--asl_anno_dir",
        type=str,
        default=None,
        help="Single ASL annotation root. Prefer YAML datasets for multi-dataset evaluation.",
    )
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        default=False,
        help="Disable image undistortion before VGGT preprocessing.",
    )


def load_asl_sequence_entries(
    dataset_name,
    data_root,
    annotation_dir,
    sequence_names=None,
    camera_names=None,
    min_num_images=24,
    split="test",
    annotation_prefix=None,
):
    if sequence_names is not None and len(sequence_names) == 0:
        return []

    data_root = Path(data_root)
    annotation_dir = Path(annotation_dir)
    raw_annotation = _load_asl_annotation(
        dataset_name=dataset_name,
        annotation_dir=annotation_dir,
        sequence_names=sequence_names,
        split=split,
        annotation_prefix=annotation_prefix,
    )
    entries = _build_euroc_sequence_entries(
        raw_annotation=raw_annotation,
        euroc_dir=data_root,
        camera_names=tuple(camera_names or []),
        min_num_images=min_num_images,
    )
    for entry in entries:
        entry["dataset_name"] = dataset_name
        entry["image_root"] = str(data_root)
    return entries


def evaluate_asl_datasets(args, model, device, dtype, predictor=None):
    dataset_results = {}
    for dataset_spec in _dataset_specs_from_args(args):
        dataset_name = _get_dataset_field(dataset_spec, "name", "dataset_name")
        sequence_names = dataset_spec.get("sequence_names")
        if sequence_names is not None and len(sequence_names) == 0:
            continue

        sequence_entries = load_asl_sequence_entries(
            dataset_name=dataset_name,
            data_root=_get_dataset_field(dataset_spec, "data_root", "asl_dir", "ASL_DIR"),
            annotation_dir=_get_dataset_field(
                dataset_spec, "annotation_dir", "asl_anno_dir", "ASL_ANNOTATION_DIR"
            ),
            sequence_names=sequence_names,
            camera_names=dataset_spec.get("camera_names", getattr(args, "camera_names", None)),
            min_num_images=getattr(args, "min_num_images", 24),
            split=dataset_spec.get("split", getattr(args, "split", "test")),
            annotation_prefix=dataset_spec.get("annotation_prefix"),
        )
        dataset_results[dataset_name] = {
            "clean": evaluate_sequences(
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
        }
    return dataset_results


def write_asl_metrics_report(args, dataset_results):
    output_dir = Path(args.metrics_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_prefix = getattr(args, "metrics_report_prefix", None) or "asl_camera_pose"
    output_path = output_dir / f"{report_prefix}_{timestamp}.txt"
    suffix = 1
    while output_path.exists():
        output_path = output_dir / f"{report_prefix}_{timestamp}_{suffix}.txt"
        suffix += 1

    lines = [
        "ASL camera pose evaluation",
        "",
        "[Run config]",
        f"config: {getattr(args, 'config', None)}",
        f"model_path: {args.model_path}",
        f"split: {args.split}",
        f"use_imu: {getattr(args, 'use_imu', False)}",
        f"imu_window_ns: {getattr(args, 'imu_window_ns', None)}",
        f"imu_num_samples: {getattr(args, 'imu_num_samples', None)}",
        f"imu_feature_order: {getattr(args, 'imu_feature_order', None)}",
        f"num_frames: {args.num_frames}",
        f"min_num_images: {args.min_num_images}",
        f"fast_eval: {args.fast_eval}",
        f"seed: {args.seed}",
        "",
    ]
    for dataset_spec in _dataset_specs_from_args(args):
        dataset_name = _get_dataset_field(dataset_spec, "name", "dataset_name")
        lines.append(f"[Dataset {dataset_name}]")
        lines.append(f"data_root: {_get_dataset_field(dataset_spec, 'data_root', 'asl_dir', 'ASL_DIR')}")
        lines.append(
            "annotation_dir: "
            f"{_get_dataset_field(dataset_spec, 'annotation_dir', 'asl_anno_dir', 'ASL_ANNOTATION_DIR')}"
        )
        lines.append(f"camera_names: {dataset_spec.get('camera_names', getattr(args, 'camera_names', None))}")
        lines.append(f"sequence_names: {dataset_spec.get('sequence_names')}")
        for variant_name, results in dataset_results.get(dataset_name, {}).items():
            lines.extend(_format_variant_lines(dataset_name, variant_name, results))
        lines.append("")
    lines.extend(_format_summary_lines(dataset_results))
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def run(args, model, device, dtype):
    dataset_results = evaluate_asl_datasets(
        args=args,
        model=model,
        device=device,
        dtype=dtype,
    )

    for dataset_name, variant_results in dataset_results.items():
        for variant_name, results in variant_results.items():
            for line in _format_variant_lines(dataset_name, variant_name, results):
                print(line)
            print("")

    for line in _format_summary_lines(dataset_results):
        print(line)

    report_path = write_asl_metrics_report(args, dataset_results)
    print(f"Saved ASL metrics report to: {report_path}")
    return dataset_results


def _load_asl_annotation(dataset_name, annotation_dir, sequence_names, split, annotation_prefix):
    manifest_path = annotation_dir / "sequence_manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as fin:
            manifest = json.load(fin)
        manifest_sequences = manifest.get("sequences", {})
        requested = list(sequence_names) if sequence_names is not None else sorted(manifest_sequences)
        annotation = {}
        for sequence_name in requested:
            manifest_key = _resolve_manifest_sequence_key(sequence_name, manifest_sequences)
            record = manifest_sequences[manifest_key]
            with gzip.open(annotation_dir / record["file"], "rt", encoding="utf-8") as fin:
                annotation.update(json.load(fin))
        return annotation

    prefix = annotation_prefix or dataset_name
    annotation_path = annotation_dir / f"{prefix}_{split}.jgz"
    with gzip.open(annotation_path, "rt", encoding="utf-8") as fin:
        return json.load(fin)


def _resolve_manifest_sequence_key(sequence_name, manifest_sequences):
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


def _dataset_specs_from_args(args):
    dataset_specs = getattr(args, "datasets", None)
    if dataset_specs is not None:
        return list(dataset_specs)
    if getattr(args, "asl_dir", None) and getattr(args, "asl_anno_dir", None):
        return [
            {
                "name": getattr(args, "dataset_name", "asl"),
                "data_root": args.asl_dir,
                "annotation_dir": args.asl_anno_dir,
                "camera_names": getattr(args, "camera_names", None),
                "sequence_names": getattr(args, "sequence_names", None),
            }
        ]
    raise ValueError("ASL evaluation requires YAML datasets or --asl_dir with --asl_anno_dir.")


def _get_dataset_field(dataset_spec, *names):
    for name in names:
        if name in dataset_spec and dataset_spec[name] is not None:
            return dataset_spec[name]
    raise KeyError(f"Dataset spec missing one of: {', '.join(names)}")


def _format_variant_lines(dataset_name, variant_name, results):
    lines = [f"ASL dataset: {dataset_name}", f"variant: {variant_name}"]
    for sequence_result in results["per_sequence"]:
        lines.append(
            f"{sequence_result['seq_name']} R_ACC@5: {sequence_result['R_ACC@5']:.4f} "
            f"T_ACC@5: {sequence_result['T_ACC@5']:.4f}"
        )
    lines.append(
        "ASL camera pose summary: "
        f"{results['AUC@30']:.4f} (AUC@30), "
        f"{results['AUC@15']:.4f} (AUC@15), "
        f"{results['AUC@5']:.4f} (AUC@5), "
        f"{results['AUC@3']:.4f} (AUC@3)"
    )
    return lines


def _format_summary_lines(dataset_results):
    lines = ["ASL camera pose dataset summary:"]
    for dataset_name, variant_results in dataset_results.items():
        for variant_name, results in variant_results.items():
            lines.append(
                f"{dataset_name}/{variant_name}: "
                f"{results['AUC@30']:.4f} (AUC@30), "
                f"{results['AUC@15']:.4f} (AUC@15), "
                f"{results['AUC@5']:.4f} (AUC@5), "
                f"{results['AUC@3']:.4f} (AUC@3), "
                f"{results['num_sequences']} sequences"
            )
    return lines
