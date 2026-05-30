from unittest.mock import patch

import torch
import torch.nn as nn
import pytest

from vggt.models.vggt import VGGT


class TinyAggregator(nn.Module):
    def __init__(self, img_size=14, patch_size=14, embed_dim=8, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.received_motion_tokens = None
        self.received_imu_fusion = None

    def forward(self, images, motion_tokens=None, imu_fusion=None):
        self.received_motion_tokens = motion_tokens
        self.received_imu_fusion = imu_fusion
        batch_size, sequence_length = images.shape[:2]
        tokens = torch.zeros(
            batch_size,
            sequence_length,
            1,
            self.embed_dim * 2,
            dtype=images.dtype,
            device=images.device,
        )
        return [tokens], 0


def test_vggt_ignores_optional_imu_fields_when_imu_disabled():
    with patch("vggt.models.vggt.Aggregator", TinyAggregator):
        model = VGGT(
            img_size=14,
            patch_size=14,
            embed_dim=8,
            enable_camera=False,
            enable_depth=False,
            enable_point=False,
            enable_track=False,
        )

    output = model(
        images=torch.randn(1, 2, 3, 14, 14),
        imu_windows=torch.randn(1, 2, 5, 6),
        imu_window_masks=torch.ones(1, 2, 5, dtype=torch.bool),
        degradation_metadata=[["clean", "clean"]],
    )

    assert output == {}
    assert model.aggregator.received_motion_tokens is None
    assert model.aggregator.received_imu_fusion is None


def test_vggt_encodes_imu_and_passes_motion_tokens_to_aggregator():
    with patch("vggt.models.vggt.Aggregator", TinyAggregator):
        model = VGGT(
            img_size=14,
            patch_size=14,
            embed_dim=8,
            enable_camera=False,
            enable_depth=False,
            enable_point=False,
            enable_track=False,
            imu={
                "enabled": True,
                "input_dim": 6,
                "hidden_dim": 16,
                "num_layers": 1,
                "num_heads": 4,
                "dropout": 0.0,
            },
            fusion={"enabled": True, "type": "film", "hidden_dim": 16},
        )

    output = model(
        images=torch.randn(1, 2, 3, 14, 14),
        imu_windows=torch.randn(1, 2, 5, 6),
        imu_window_masks=torch.ones(1, 2, 5, dtype=torch.bool),
    )

    assert output["motion_tokens"].shape == (1, 2, 8)
    assert output["motion_risk"].shape == (1, 2, 1)
    assert model.aggregator.received_motion_tokens is output["motion_tokens"]
    assert model.aggregator.received_imu_fusion is model.imu_fusion


def test_vggt_rejects_unsupported_fusion_insert_at():
    with patch("vggt.models.vggt.Aggregator", TinyAggregator):
        with pytest.raises(ValueError, match="Unsupported fusion.insert_at"):
            VGGT(
                img_size=14,
                patch_size=14,
                embed_dim=8,
                enable_camera=False,
                enable_depth=False,
                enable_point=False,
                enable_track=False,
                imu={"enabled": True},
                fusion={"enabled": True, "type": "film", "insert_at": "after_attention"},
            )
