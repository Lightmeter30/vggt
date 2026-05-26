import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from launch import parse_args


def test_launch_accepts_hydra_overrides():
    args = parse_args(
        [
            "--config",
            "euroc_imu_film",
            "mode=val",
            "checkpoint.resume_checkpoint_path=logs/checkpoint.pt",
        ]
    )

    assert args.config == "euroc_imu_film"
    assert args.overrides == [
        "mode=val",
        "checkpoint.resume_checkpoint_path=logs/checkpoint.pt",
    ]
