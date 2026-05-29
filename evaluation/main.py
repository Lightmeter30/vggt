import argparse
import sys
from pathlib import Path
from types import SimpleNamespace


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
    parser.add_argument("--config", type=str, default=None, help="Path to an evaluation YAML config.")
    parser.add_argument("--dataset", default=None, help="Dataset to evaluate, e.g. euroc.")
    parser.add_argument("--task", default=None, help="Task to evaluate, e.g. camera_pose.")
    parser.add_argument("--model_path", default=None, help="Path to a VGGT checkpoint.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override.")
    return parser


def parse_args(argv=None):
    parser = build_base_parser()
    known_args, _ = parser.parse_known_args(argv)
    config = _load_yaml_config(known_args.config) if known_args.config else {}
    dataset, task = _resolve_dataset_task(known_args, config)
    if dataset is None or task is None:
        parser.error("--dataset and --task are required unless --config supplies them.")

    try:
        evaluator = get_evaluator(dataset, task)
    except ValueError as exc:
        parser.error(str(exc))
    evaluator.add_arguments(parser)
    cli_args = parser.parse_args(argv)
    if known_args.config:
        args = _merge_config_and_cli_args(config, cli_args, parser, argv, dataset, task)
    else:
        args = cli_args

    if args.model_path is None:
        parser.error("--model_path is required unless --config supplies it.")
    if args.seed is None:
        args.seed = 0
    args.dataset = dataset if args.dataset is None else args.dataset
    args.task = task if args.task is None else args.task
    return args


def main(argv=None):
    args = parse_args(argv)
    set_random_seeds(args.seed)
    evaluator = get_evaluator(args.dataset, args.task)
    should_load_model = getattr(evaluator, "should_load_model", lambda parsed_args: True)
    if should_load_model(args):
        device = resolve_device(args.device)
        dtype = resolve_inference_dtype(device)
        model = load_model(device, args.model_path)
    else:
        device = None
        dtype = None
        model = None
    evaluator.run(args, model=model, device=device, dtype=dtype)


def _load_yaml_config(config_path):
    import yaml

    with open(config_path, "r", encoding="utf-8") as fin:
        config = yaml.safe_load(fin) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Evaluation config must be a mapping: {config_path}")
    return config


def _resolve_dataset_task(known_args, config):
    dataset = known_args.dataset or config.get("dataset")
    if dataset is None and "datasets" in config:
        dataset = "asl"
    task = known_args.task or config.get("task")
    return dataset, task


def _merge_config_and_cli_args(config, cli_args, parser, argv, dataset, task):
    merged = _parser_defaults(parser)
    merged.update(config)
    merged["config"] = cli_args.config
    merged["dataset"] = dataset
    merged["task"] = task

    if "undistort_images" in merged and "no_undistort" not in config:
        merged["no_undistort"] = not bool(merged["undistort_images"])

    provided_dests = _provided_option_dests(parser, argv)
    for dest in provided_dests:
        merged[dest] = getattr(cli_args, dest)
    return SimpleNamespace(**merged)


def _parser_defaults(parser):
    defaults = {}
    for action in parser._actions:
        if action.dest == argparse.SUPPRESS:
            continue
        if action.default == argparse.SUPPRESS:
            continue
        defaults[action.dest] = action.default
    return defaults


def _provided_option_dests(parser, argv):
    if argv is None:
        argv = sys.argv[1:]
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    provided = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        if option in option_to_dest:
            provided.add(option_to_dest[option])
    return provided


if __name__ == "__main__":
    main()
