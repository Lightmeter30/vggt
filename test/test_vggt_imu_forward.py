from unittest.mock import patch
from types import SimpleNamespace

import torch
import torch.nn as nn
import pytest

from vggt.models.vggt import VGGT


class TinyAggregator(nn.Module):
    def __init__(self, img_size=14, patch_size=14, embed_dim=8, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_start_idx = 0
        self.received_token_state = None

    def prepare_tokens(self, images):
        batch_size, sequence_length = images.shape[:2]
        tokens = torch.zeros(
            batch_size * sequence_length,
            1,
            self.embed_dim,
            dtype=images.dtype,
            device=images.device,
        )
        return SimpleNamespace(
            tokens=tokens,
            batch_size=batch_size,
            sequence_length=sequence_length,
            patch_token_count=1,
        )

    def aggregate_tokens(self, token_state, attention_context_provider=None):
        del attention_context_provider
        self.received_token_state = token_state
        batch_size = token_state.batch_size
        sequence_length = token_state.sequence_length
        tokens = torch.zeros(
            batch_size,
            sequence_length,
            1,
            self.embed_dim * 2,
            dtype=token_state.tokens.dtype,
            device=token_state.tokens.device,
        )
        return [tokens], 0

    def forward(self, images, attention_context_provider=None):
        return self.aggregate_tokens(
            self.prepare_tokens(images),
            attention_context_provider=attention_context_provider,
        )


class AddOneFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.received_motion_tokens = None

    def forward(
        self,
        tokens,
        motion_tokens,
        patch_start_idx,
        batch_size,
        sequence_length,
        patch_token_count=None,
    ):
        del patch_start_idx, batch_size, sequence_length, patch_token_count
        self.received_motion_tokens = motion_tokens
        return tokens + 1.0


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
    assert model.imu_encoder is None
    assert model.imu_fusion is None
    assert model.aggregator.received_token_state is not None


def test_vggt_encodes_imu_and_fuses_before_aggregation():
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

    model.imu_fusion = AddOneFusion()
    output = model(
        images=torch.randn(1, 2, 3, 14, 14),
        imu_windows=torch.randn(1, 2, 5, 6),
        imu_window_masks=torch.ones(1, 2, 5, dtype=torch.bool),
    )

    assert output["motion_tokens"].shape == (1, 2, 8)
    assert output["motion_risk"].shape == (1, 2, 1)
    assert model.imu_fusion.received_motion_tokens is output["motion_tokens"]
    assert model.aggregator.received_token_state is not None
    assert torch.equal(
        model.aggregator.received_token_state.tokens,
        torch.ones_like(model.aggregator.received_token_state.tokens),
    )


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
