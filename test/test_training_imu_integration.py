import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from trainer import Trainer


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.received_kwargs = None

    def forward(self, **kwargs):
        self.received_kwargs = kwargs
        return {"dummy": torch.tensor(1.0)}


class DummyLoss(nn.Module):
    def forward(self, predictions, batch):
        return {"objective": predictions["dummy"] * 0.0}


def test_trainer_step_passes_optional_imu_fields_without_degradation_metadata():
    trainer = object.__new__(Trainer)
    trainer.loss = DummyLoss()
    trainer.steps = {"train": 0}
    trainer._update_and_log_scalars = lambda *args, **kwargs: None
    trainer._log_tb_visuals = lambda *args, **kwargs: None

    model = RecordingModel()
    batch = {
        "images": torch.randn(2, 3, 3, 14, 14),
        "imu_windows": torch.randn(2, 3, 5, 6),
        "imu_window_masks": torch.ones(2, 3, 5, dtype=torch.bool),
    }

    trainer._step(batch, model, "train", {})

    assert model.received_kwargs["images"] is batch["images"]
    assert model.received_kwargs["imu_windows"] is batch["imu_windows"]
    assert model.received_kwargs["imu_window_masks"] is batch["imu_window_masks"]
    assert "degradation_metadata" not in model.received_kwargs


def test_load_checkpoint_in_val_mode_ignores_optimizer_state(tmp_path):
    source_model = nn.Linear(1, 1)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": source_model.state_dict(),
            "optimizer": {"state": {}, "param_groups": []},
            "steps": {"train": 5, "val": 0},
            "time_elapsed": 12.0,
        },
        checkpoint_path,
    )

    trainer = object.__new__(Trainer)
    trainer.rank = 0
    trainer.mode = "val"
    trainer.model = nn.Linear(1, 1)
    trainer.optim_conf = SimpleNamespace(amp=SimpleNamespace(enabled=False))
    trainer.checkpoint_conf = SimpleNamespace(strict=True)

    trainer._load_resuming_checkpoint(str(checkpoint_path))

    assert trainer.steps == {"train": 5, "val": 0}
    assert trainer.ckpt_time_elapsed == 12.0


def test_update_scalars_logs_objective_as_loss_objective():
    class RecordingMeter:
        def __init__(self):
            self.value = None
            self.count = None

        def update(self, value, count):
            self.value = value
            self.count = count

    trainer = object.__new__(Trainer)
    trainer.rank = 1
    trainer.logging_conf = SimpleNamespace(log_freq=1)
    trainer._get_scalar_log_keys = lambda phase: ["loss_objective"]

    meter = RecordingMeter()
    trainer._update_and_log_scalars(
        {
            "objective": torch.tensor(2.5),
            "extrinsics": torch.zeros(3, 2, 3, 4),
        },
        "val",
        0,
        {"Loss/val_loss_objective": meter},
    )

    assert meter.value == 2.5
    assert meter.count == 3
