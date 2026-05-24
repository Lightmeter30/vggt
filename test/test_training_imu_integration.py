import sys
from pathlib import Path

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


def test_trainer_step_passes_optional_imu_and_degradation_fields():
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
        "degradation_metadata": [["clean"] * 3] * 2,
    }

    trainer._step(batch, model, "train", {})

    assert model.received_kwargs["images"] is batch["images"]
    assert model.received_kwargs["imu_windows"] is batch["imu_windows"]
    assert model.received_kwargs["imu_window_masks"] is batch["imu_window_masks"]
    assert model.received_kwargs["degradation_metadata"] is batch["degradation_metadata"]

