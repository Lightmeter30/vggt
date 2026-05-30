import torch

from vggt.layers.visual_imu_fusion import VisualIMUFiLM


def test_visual_imu_film_is_identity_at_initialization():
    batch_size = 2
    sequence_length = 3
    token_count = 7
    embed_dim = 8
    patch_start_idx = 2
    tokens = torch.randn(batch_size * sequence_length, token_count, embed_dim)
    motion_tokens = torch.randn(batch_size, sequence_length, embed_dim)
    fusion = VisualIMUFiLM(embed_dim=embed_dim, hidden_dim=16)

    fused = fusion(
        tokens=tokens,
        motion_tokens=motion_tokens,
        patch_start_idx=patch_start_idx,
        batch_size=batch_size,
        sequence_length=sequence_length,
        patch_token_count=token_count - patch_start_idx,
    )

    assert torch.allclose(fused, tokens)


def test_visual_imu_film_preserves_special_tokens_when_active():
    batch_size = 1
    sequence_length = 2
    token_count = 5
    embed_dim = 4
    patch_start_idx = 2
    tokens = torch.randn(batch_size * sequence_length, token_count, embed_dim)
    motion_tokens = torch.ones(batch_size, sequence_length, embed_dim)
    fusion = VisualIMUFiLM(embed_dim=embed_dim, hidden_dim=8)
    with torch.no_grad():
        fusion.film[-1].bias[:embed_dim].fill_(0.5)
        fusion.film[-1].bias[embed_dim:].fill_(0.25)

    fused = fusion(
        tokens=tokens,
        motion_tokens=motion_tokens,
        patch_start_idx=patch_start_idx,
        batch_size=batch_size,
        sequence_length=sequence_length,
        patch_token_count=token_count - patch_start_idx,
    )

    assert torch.allclose(fused[:, :patch_start_idx], tokens[:, :patch_start_idx])
    assert not torch.allclose(fused[:, patch_start_idx:], tokens[:, patch_start_idx:])
