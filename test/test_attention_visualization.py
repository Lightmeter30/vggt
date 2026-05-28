from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from PIL import Image

from vggt.layers.attention import Attention
from vggt.models.vggt import VGGT
from vggt.utils.attention_visualization import (
    AttentionCaptureConfig,
    AttentionCaptureSession,
    AttentionRecord,
)


def test_attention_capture_records_selected_query_weights_sum_to_one(tmp_path):
    torch.manual_seed(7)
    attention = Attention(dim=4, num_heads=2, fused_attn=True)
    session = AttentionCaptureSession(
        AttentionCaptureConfig(
            output_dir=tmp_path,
            block_indices=[0],
            query_frames=["first"],
            query_kinds=["camera"],
        )
    )
    context = session.make_context(
        block_index=0,
        attention_type="global",
        batch_size=1,
        sequence_length=2,
        patch_start_idx=1,
        patch_grid=(1, 2),
        token_count=6,
    )

    x = torch.randn(1, 6, 4)
    _ = attention(x, attn_context=context)

    assert len(session.records) == 1
    record = session.records[0]
    assert record.block_index == 0
    assert record.query_name == "frame_000_camera"
    assert record.query_token_index == 0
    assert record.key_attention.shape == (6,)
    assert record.patch_attention.shape == (2, 1, 2)
    assert torch.allclose(record.key_attention.sum(), torch.tensor(1.0), atol=1e-5)


def test_attention_capture_writes_overlay_pngs_and_manifest(tmp_path):
    image_paths = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0))):
        image_path = tmp_path / f"image_{index:03d}.png"
        Image.new("RGB", (8, 4), color=color).save(image_path)
        image_paths.append(str(image_path))

    input_images = torch.zeros(2, 3, 4, 8)
    input_images[0, 0] = 1.0
    input_images[1, 1] = 1.0
    session = AttentionCaptureSession(
        AttentionCaptureConfig(output_dir=tmp_path / "attention", block_indices=[0])
    )
    session.records.append(
        AttentionRecord(
            block_index=0,
            query_name="frame_000_camera",
            query_token_index=0,
            key_attention=torch.full((6,), 1.0 / 6.0),
            patch_attention=torch.tensor(
                [
                    [[0.0, 1.0]],
                    [[1.0, 0.0]],
                ]
            ),
            patch_grid=(1, 2),
            sequence_length=2,
            patch_start_idx=1,
        )
    )

    run_dir = session.write_outputs(
        checkpoint_path="ckpt/model.pt",
        image_paths=image_paths,
        input_images=input_images,
        preprocess_mode="crop",
    )

    manifest_path = run_dir / "manifest.json"
    assert manifest_path.is_file()
    assert (run_dir / "block_000" / "query_frame_000_camera" / "frame_000_image_000.png").is_file()
    assert (run_dir / "block_000" / "query_frame_000_camera" / "frame_001_image_001.png").is_file()
    assert "frame_000_camera" in manifest_path.read_text(encoding="utf-8")


class TinyAggregator(nn.Module):
    def __init__(self, img_size=14, patch_size=14, embed_dim=8, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.received_attention_capture = "unset"

    def forward(self, images, motion_tokens=None, imu_fusion=None, attention_capture=None):
        del motion_tokens, imu_fusion
        self.received_attention_capture = attention_capture
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


def test_vggt_forward_passes_attention_capture_to_aggregator():
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

    images = torch.randn(1, 2, 3, 14, 14)
    assert model(images) == {}
    assert model.aggregator.received_attention_capture is None

    attention_capture = object()
    assert model(images, attention_capture=attention_capture) == {}
    assert model.aggregator.received_attention_capture is attention_capture
