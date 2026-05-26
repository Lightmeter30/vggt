import torch
import torch.nn as nn


class IMUEncoder(nn.Module):
    """
    输入: IMU 窗口数据 [B, S, T, 6] (6 维 = 加速度计 3 维 + 陀螺仪 3 维)
    结构: Linear投影 → 时间位置编码(MLP) → TransformerEncoder(2层) → 均值池化 → 输出投影
    输出: 运动特征 [B, S, embed_dim] 和 运动风险 [0, 1] 标量, 通过 Sigmoid 预测该帧运动是否"危险"（如快速旋转/模糊）
    """
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 256,
        embed_dim: int = 1024,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.output_proj = nn.Linear(hidden_dim, embed_dim)
        self.risk_head = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        imu_windows: torch.Tensor,
        imu_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if imu_windows.ndim != 4:
            raise ValueError(
                f"Expected imu_windows with shape [B,S,T,{self.input_dim}], got {tuple(imu_windows.shape)}"
            )
        if imu_windows.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected IMU feature dim {self.input_dim}, got {imu_windows.shape[-1]}"
            )

        batch_size, sequence_length, sample_count, _ = imu_windows.shape
        flat_imu = imu_windows.reshape(batch_size * sequence_length, sample_count, self.input_dim)

        if imu_masks is None:
            flat_masks = torch.ones(
                batch_size * sequence_length,
                sample_count,
                dtype=torch.bool,
                device=imu_windows.device,
            )
        else:
            if imu_masks.shape != (batch_size, sequence_length, sample_count):
                raise ValueError(
                    "Expected imu_masks with shape "
                    f"{(batch_size, sequence_length, sample_count)}, got {tuple(imu_masks.shape)}"
                )
            flat_masks = imu_masks.reshape(batch_size * sequence_length, sample_count).to(torch.bool)

        relative_time = torch.linspace(
            -1.0,
            1.0,
            steps=sample_count,
            device=imu_windows.device,
            dtype=imu_windows.dtype,
        ).view(1, sample_count, 1)
        tokens = self.input_proj(flat_imu) + self.time_mlp(relative_time)

        key_padding_mask = ~flat_masks
        all_invalid = key_padding_mask.all(dim=1)
        safe_key_padding_mask = key_padding_mask.clone()
        if all_invalid.any():
            safe_key_padding_mask[all_invalid] = False

        encoded = self.temporal_encoder(
            tokens,
            src_key_padding_mask=safe_key_padding_mask,
        )
        mask_float = flat_masks.to(encoded.dtype).unsqueeze(-1)
        valid_counts = mask_float.sum(dim=1).clamp_min(1.0)
        pooled = (encoded * mask_float).sum(dim=1) / valid_counts
        if all_invalid.any():
            pooled = pooled.masked_fill(all_invalid.unsqueeze(-1), 0.0)

        motion_tokens = self.output_proj(pooled).view(batch_size, sequence_length, self.embed_dim)
        motion_risk = self.risk_head(motion_tokens)
        return motion_tokens, motion_risk
