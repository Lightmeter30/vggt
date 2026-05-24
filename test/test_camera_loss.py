import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = REPO_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from loss import compute_camera_loss
from vggt.utils.pose_enc import extri_intri_to_pose_encoding


def test_camera_loss_can_ignore_sparse_point_masks_for_euroc_camera_only():
    batch_size = 1
    sequence_length = 2
    height = 8
    width = 8
    extrinsics = torch.eye(4)[:3].view(1, 1, 3, 4).repeat(
        batch_size, sequence_length, 1, 1
    )
    intrinsics = torch.eye(3).view(1, 1, 3, 3).repeat(
        batch_size, sequence_length, 1, 1
    )
    images = torch.zeros(batch_size, sequence_length, 3, height, width)
    point_masks = torch.zeros(batch_size, sequence_length, height, width, dtype=torch.bool)
    gt_pose = extri_intri_to_pose_encoding(extrinsics, intrinsics, (height, width))
    predictions = {"pose_enc_list": [gt_pose + 0.1]}
    batch = {
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "images": images,
        "point_masks": point_masks,
    }

    masked_loss = compute_camera_loss(
        predictions,
        batch,
        weight_trans=1.0,
        weight_rot=1.0,
        weight_focal=1.0,
        use_point_mask=True,
    )
    unmasked_loss = compute_camera_loss(
        predictions,
        batch,
        weight_trans=1.0,
        weight_rot=1.0,
        weight_focal=1.0,
        use_point_mask=False,
    )

    assert masked_loss["loss_camera"].item() == 0.0
    assert unmasked_loss["loss_camera"].item() > 0.0

