#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import torch.nn as nn

from pyskl.models.recognizers import Recognizer3D


# -------------------------------------------------------------
# PoseConv3D
# -------------------------------------------------------------
class PoseConv3D(Recognizer3D):
    """
    https://github.com/kennymckormick/pyskl/blob/main/pyskl/models/cnns/resnet3d.py
    https://github.com/kennymckormick/pyskl/blob/main/pyskl/models/cnns/rgbposeconv3d.py
    """

    def __init__(
        self,
        backbone=dict(
            type="ResNet3dSlowOnly",
            in_channels=17,
            base_channels=32,
            num_stages=3,
            out_indices=(2,),
            stage_blocks=(4, 6, 3),
            conv1_stride=(1, 1),
            pool1_stride=(1, 1),
            inflate=(0, 1, 1),
            spatial_strides=(2, 2, 2),
            temporal_strides=(1, 1, 2),
        ),
        cls_head=None,
        train_cfg=dict(),
        test_cfg=dict(),
    ):
        super().__init__(backbone, cls_head, train_cfg, test_cfg)

        self.cls_head = nn.Identity()
        self.final_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, imgs):
        if imgs.shape[1] == 1:
            imgs = imgs.squeeze(1)
        return self.final_pool(self.backbone(imgs)).squeeze()


# -------------------------------------------------------------
# Conv3DAdapter
# -------------------------------------------------------------
class Conv3DAdapter(nn.Module):
    def __init__(self, in_channels=21, out_channels=17):
        super().__init__()
        self.pose = nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        return self.pose(x.squeeze(1))
