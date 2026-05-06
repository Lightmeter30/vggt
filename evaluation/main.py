import argparse
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.common.model import (  # noqa: E402
    load_model,
    resolve_device,
    resolve_inference_dtype,
    set_random_seeds,
)
from evaluation.tasks import get_evaluator  # noqa: E402


def build_base_parser():
    parser = argparse.ArgumentParser(description="VGGT evaluation entrypoint.")
    parser.add_argument("--dataset", required=True, help="Dataset to evaluate, e.g. euroc.")
    parser.add_argument("--task", required=True, help="Task to evaluate, e.g. camera_pose.")
    parser.add_argument("--model_path", required=True, help="Path to a VGGT checkpoint.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override.")
    return parser


def parse_args(argv=None):
    parser = build_base_parser()
    known_args, _ = parser.parse_known_args(argv)
    try:
        evaluator = get_evaluator(known_args.dataset, known_args.task)
    except ValueError as exc:
        parser.error(str(exc))
    evaluator.add_arguments(parser)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_random_seeds(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_inference_dtype(device)
    model = load_model(device, args.model_path)
    evaluator = get_evaluator(args.dataset, args.task)
    evaluator.run(args, model=model, device=device, dtype=dtype)


if __name__ == "__main__":
    main()
