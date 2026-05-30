from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from PIL import Image

from vggt.layers.attention import Attention
from vggt.models.aggregator import Aggregator
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


def test_aggregator_uses_attention_context_provider_hook():
    aggregator = Aggregator(
        img_size=14,
        patch_size=14,
        embed_dim=8,
        depth=1,
        num_heads=2,
        mlp_ratio=1.0,
        num_register_tokens=0,
        patch_embed="conv",
        aa_order=["frame", "global"],
        qk_norm=False,
        rope_freq=-1,
    )
    aggregator.eval()
    calls = []

    def context_provider(**metadata):
        calls.append(metadata)
        return None

    images = torch.randn(1, 2, 3, 14, 14)
    with torch.no_grad():
        aggregated_tokens, patch_start_idx = aggregator(
            images,
            attention_context_provider=context_provider,
        )

    assert len(aggregated_tokens) == 1
    assert patch_start_idx == 1
    assert calls[0]["attention_type"] == "global"
    assert calls[0]["sequence_length"] == 2


class TinyAggregator(nn.Module):
    def __init__(self, img_size=14, patch_size=14, embed_dim=8, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.received_attention_context_provider = "unset"
        self.patch_start_idx = 0

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
        self.received_attention_context_provider = attention_context_provider
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


def test_vggt_forward_passes_attention_context_provider_to_aggregator():
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
    assert model.aggregator.received_attention_context_provider is None

    attention_capture = AttentionCaptureSession(
        AttentionCaptureConfig(output_dir=Path("/tmp/unused_attention_test"))
    )
    assert model(images, attention_capture=attention_capture) == {}
    assert callable(model.aggregator.received_attention_context_provider)
