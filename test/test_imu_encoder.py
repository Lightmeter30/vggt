import torch

from vggt.models.imu_encoder import IMUEncoder


def test_imu_encoder_returns_motion_token_and_risk_with_masks():
    encoder = IMUEncoder(
        input_dim=6,
        hidden_dim=16,
        embed_dim=12,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    imu_windows = torch.randn(2, 3, 5, 6)
    imu_masks = torch.ones(2, 3, 5, dtype=torch.bool)
    imu_masks[0, 1] = False

    motion_tokens, motion_risk = encoder(imu_windows, imu_masks)

    assert motion_tokens.shape == (2, 3, 12)
    assert motion_risk.shape == (2, 3, 1)
    assert torch.isfinite(motion_tokens).all()
    assert torch.isfinite(motion_risk).all()
    assert torch.all((motion_risk >= 0.0) & (motion_risk <= 1.0))


def test_imu_encoder_accepts_missing_mask_as_all_valid():
    encoder = IMUEncoder(
        input_dim=6,
        hidden_dim=16,
        embed_dim=12,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    imu_windows = torch.randn(1, 2, 4, 6)

    motion_tokens, motion_risk = encoder(imu_windows)

    assert motion_tokens.shape == (1, 2, 12)
    assert motion_risk.shape == (1, 2, 1)

