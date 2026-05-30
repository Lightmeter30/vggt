import torch
import torch.nn as nn


class VisualIMUFiLM(nn.Module):
    """
    视觉-IMU 融合模块，使用 FiLM（Feature-wise Linear Modulation）机制将运动特征调制到视觉特征上。
    输入:
        - tokens: 视觉特征 [B*S, P, C]，其中 P 是每帧的 token 数量（包括特殊 token 和 patch token），C 是特征维度（应与 embed_dim 匹配）
        - motion_tokens: 运动特征 [B, S, embed_dim]，来自 IMUEncoder 的输出
        - patch_start_idx: 视觉 token 中 patch token 的起始索引，之前的 token 被视为特殊 token（如 CLS token）
        - batch_size: 批量大小 B
        - sequence_length: 序列长度 S
        - patch_token_count: 可选的 patch token 数量，如果提供则会验证 tokens 中 patch token 的数量是否正确
    输出:
        - modulated_tokens: 调制后的视觉特征 [B*S, P, C]
    """
    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int | None = None,
        zero_init_gamma_scale: float = 1.0,
        zero_init_beta_scale: float = 1.0,
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        hidden_dim = hidden_dim or embed_dim * 2
        self.embed_dim = embed_dim
        self.zero_init_gamma_scale = float(zero_init_gamma_scale)
        self.zero_init_beta_scale = float(zero_init_beta_scale)
        self.film = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim * 2),
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def forward(
        self,
        tokens: torch.Tensor,
        motion_tokens: torch.Tensor,
        patch_start_idx: int,
        batch_size: int,
        sequence_length: int,
        patch_token_count: int | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens with shape [B*S,P,C], got {tuple(tokens.shape)}")
        if motion_tokens.shape != (batch_size, sequence_length, self.embed_dim):
            raise ValueError(
                "Expected motion_tokens with shape "
                f"{(batch_size, sequence_length, self.embed_dim)}, got {tuple(motion_tokens.shape)}"
            )
        if tokens.shape[0] != batch_size * sequence_length:
            raise ValueError(
                f"Expected first token dim {batch_size * sequence_length}, got {tokens.shape[0]}"
            )
        if tokens.shape[-1] != self.embed_dim:
            raise ValueError(f"Expected token dim {self.embed_dim}, got {tokens.shape[-1]}")
        if patch_start_idx < 0 or patch_start_idx > tokens.shape[1]:
            raise ValueError(f"Invalid patch_start_idx: {patch_start_idx}")
        if patch_token_count is not None and tokens.shape[1] - patch_start_idx != patch_token_count:
            raise ValueError(
                f"Expected {patch_token_count} patch tokens, got {tokens.shape[1] - patch_start_idx}"
            )

        flat_motion = motion_tokens.reshape(batch_size * sequence_length, self.embed_dim)
        gamma_beta = self.film(flat_motion)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1) * self.zero_init_gamma_scale
        beta = beta.unsqueeze(1) * self.zero_init_beta_scale

        special_tokens = tokens[:, :patch_start_idx]
        patch_tokens = tokens[:, patch_start_idx:]
        modulated_patch_tokens = patch_tokens * (1.0 + gamma) + beta
        return torch.cat([special_tokens, modulated_patch_tokens], dim=1)
