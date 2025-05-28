#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import torch.nn as nn

from pyskl.models.recognizers import MMRecognizer3D


# -------------------------------------------------------------
# RGBPoseConv3D
# -------------------------------------------------------------
class RGBPoseConv3D(MMRecognizer3D):
    """
    https://github.com/kennymckormick/pyskl/blob/main/pyskl/models/cnns/rgbposeconv3d.py
    """

    def __init__(
        self,
        # backbone
        backbone=dict(
            type="RGBPoseConv3D",
            speed_ratio=4,
            channel_ratio=4,
            rgb_pathway=dict(  # rgb_pathway
                num_stages=4,
                lateral=True,
                lateral_infl=1,
                lateral_activate=[0, 0, 1, 1],
                base_channels=64,
                conv1_kernel=(1, 7, 7),
                inflate=(0, 0, 1, 1),
            ),
            pose_pathway=dict(  # pose_pathway
                num_stages=3,
                stage_blocks=(4, 6, 3),
                lateral=True,
                lateral_inv=True,
                lateral_infl=16,
                lateral_activate=(0, 1, 1),
                in_channels=17,
                base_channels=32,
                out_indices=(2,),
                conv1_kernel=(1, 7, 7),
                conv1_stride=(1, 1),
                pool1_stride=(1, 1),
                inflate=(0, 1, 1),
                spatial_strides=(2, 2, 2),
                temporal_strides=(1, 1, 1),
            ),
        ),
        # cls_head
        cls_head=None,
        train_cfg=dict(),
        test_cfg=dict(),
    ):
        super().__init__(backbone, cls_head, train_cfg, test_cfg)

        self.cls_head = nn.Identity()
        self.final_pool = nn.AdaptiveAvgPool3d(1)

    def _forward(self, imgs, heatmap_imgs):
        assert not hasattr(self, "neck")
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        if heatmap_imgs.shape[1] == 1:
            heatmap_imgs = heatmap_imgs.squeeze(1)

        x_rgb, x_pose = self.backbone(imgs, heatmap_imgs)

        x_rgb = self.final_pool(x_rgb).squeeze()
        x_pose = self.final_pool(x_pose).squeeze()

        return {"e_rgb": x_rgb, "e_pose": x_pose}

    def forward(self, sample):
        return self._forward(sample["imgs"], sample["heatmap_imgs"])


# -------------------------------------------------------------
# Conv3DAdapter
# -------------------------------------------------------------
class Conv3DAdapter(nn.Module):
    def __init__(self, in_channels=21, out_channels=17):
        super().__init__()
        self.pose = nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, sample):
        sample["heatmap_imgs"] = self.pose(sample["heatmap_imgs"].squeeze(1))
        return sample
