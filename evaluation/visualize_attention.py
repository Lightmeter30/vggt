import argparse
import sys
from pathlib import Path

import torch


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.common.model import load_model, resolve_device, resolve_inference_dtype, set_random_seeds  # noqa: E402
from vggt.utils.attention_visualization import (  # noqa: E402
    AttentionCaptureConfig,
    AttentionCaptureSession,
    parse_int_list,
    parse_str_list,
)
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_parser():
    parser = argparse.ArgumentParser(description="Visualize VGGT global attention maps for an image sequence.")
    parser.add_argument("--image_dir", type=Path, required=True, help="Directory containing input images.")
    parser.add_argument("--model_path", required=True, help="Path to a VGGT checkpoint.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory where attention maps will be written.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional maximum number of images to process.")
    parser.add_argument(
        "--blocks",
        type=parse_int_list,
        default=parse_int_list("0,4,11,17,23"),
        help="Comma-separated global block indices to capture.",
    )
    parser.add_argument(
        "--query_frames",
        type=parse_str_list,
        default=parse_str_list("first,middle,last"),
        help="Comma-separated query frame specs: first,middle,last,integer.",
    )
    parser.add_argument(
        "--query_kinds",
        type=parse_str_list,
        default=parse_str_list("camera,center_patch"),
        help="Comma-separated query kinds. Supported: camera,center_patch.",
    )
    parser.add_argument(
        "--preprocess_mode",
        choices=("crop", "pad"),
        default="crop",
        help="Image preprocessing mode passed to load_and_preprocess_images.",
    )
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--overlay_alpha", type=float, default=0.5, help="Heatmap overlay alpha in [0, 1].")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    set_random_seeds(args.seed)

    image_paths = _collect_image_paths(args.image_dir)
    if args.max_frames is not None:
        image_paths = image_paths[: args.max_frames]
    if not image_paths:
        raise ValueError(f"No images found in {args.image_dir}.")

    device = resolve_device(args.device)
    dtype = resolve_inference_dtype(device)
    model = load_model(device, args.model_path)
    if getattr(model, "imu_enabled", False):
        raise ValueError(
            "The loaded checkpoint enables IMU-FiLM, but evaluation/visualize_attention.py "
            "does not auto-build IMU windows yet. Use an image-only checkpoint or extend the CLI "
            "to pass imu_windows explicitly."
        )

    images = load_and_preprocess_images([str(path) for path in image_paths], mode=args.preprocess_mode).to(device)
    attention_capture = AttentionCaptureSession(
        AttentionCaptureConfig(
            output_dir=args.output_dir,
            block_indices=args.blocks,
            query_frames=args.query_frames,
            query_kinds=args.query_kinds,
            overlay_alpha=args.overlay_alpha,
        )
    )

    with torch.no_grad():
        if device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                model(images, attention_capture=attention_capture)
        else:
            model(images, attention_capture=attention_capture)

    run_dir = attention_capture.write_outputs(
        checkpoint_path=args.model_path,
        image_paths=[str(path) for path in image_paths],
        input_images=images,
        preprocess_mode=args.preprocess_mode,
    )
    print(f"Saved attention visualizations to: {run_dir}")
    print(f"Captured {len(attention_capture.records)} attention query records.")
    return run_dir


def _collect_image_paths(image_dir: Path):
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise ValueError(f"image_dir does not exist or is not a directory: {image_dir}")
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


if __name__ == "__main__":
    main()
