# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from collections.abc import Mapping
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.models.imu_encoder import IMUEncoder
from vggt.models.visual_imu_fusion import VisualIMUFiLM


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True,
                 imu=None, fusion=None, attention_bias=None, degradation_reweight=None):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.imu_encoder = None
        self.imu_fusion = None

        self.imu_enabled = bool(_config_get(imu, "enabled", False))
        if self.imu_enabled:
            self.imu_encoder = IMUEncoder(
                input_dim=int(_config_get(imu, "input_dim", 6)),
                hidden_dim=int(_config_get(imu, "hidden_dim", 256)),
                embed_dim=embed_dim,
                num_layers=int(_config_get(imu, "num_layers", 2)),
                num_heads=int(_config_get(imu, "num_heads", 4)),
                dropout=float(_config_get(imu, "dropout", 0.1)),
            )

        fusion_enabled = bool(_config_get(fusion, "enabled", False))
        if fusion_enabled:
            fusion_type = str(_config_get(fusion, "type", "film"))
            if fusion_type != "film":
                raise ValueError(f"Unsupported fusion type: {fusion_type}")
            self.imu_fusion = VisualIMUFiLM(
                embed_dim=embed_dim,
                hidden_dim=_config_get(fusion, "hidden_dim", None),
                zero_init_gamma_scale=float(_config_get(fusion, "zero_init_gamma_scale", 1.0)),
                zero_init_beta_scale=float(_config_get(fusion, "zero_init_beta_scale", 1.0)),
            )

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
        imu_windows: torch.Tensor = None,
        imu_window_masks: torch.Tensor = None,
        degradation_metadata=None,
        attention_capture=None,
    ):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None
            attention_capture (optional): Capture session used by visualization tools to record global attention maps.

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        motion_tokens = None
        motion_risk = None
        if self.imu_encoder is not None:
            if imu_windows is None:
                raise ValueError("imu.enabled=True requires imu_windows in VGGT.forward().")
            if imu_windows.ndim == 3:
                imu_windows = imu_windows.unsqueeze(0)
            if imu_window_masks is not None and imu_window_masks.ndim == 2:
                imu_window_masks = imu_window_masks.unsqueeze(0)
            motion_tokens, motion_risk = self.imu_encoder(imu_windows, imu_window_masks)

        aggregator_kwargs = {
            "motion_tokens": motion_tokens,
            "imu_fusion": self.imu_fusion,
        }
        if attention_capture is not None:
            aggregator_kwargs["attention_capture"] = attention_capture
        aggregated_tokens_list, patch_start_idx = self.aggregator(images, **aggregator_kwargs)

        predictions = {}
        if motion_tokens is not None:
            predictions["motion_tokens"] = motion_tokens
            predictions["motion_risk"] = motion_risk

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions


def _config_get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)
