# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="default",
        help="Name of the config file (without .yaml extension, default: default)"
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-style dotlist overrides, for example mode=val",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config, overrides=args.overrides)

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()

