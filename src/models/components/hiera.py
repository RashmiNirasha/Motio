#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import torch
import torch.nn as nn
from torchvision.ops import roi_align


class Hiera(nn.Module):
    """
    https://github.com/facebookresearch/hiera
    """

    def __init__(self, cfg):
        super().__init__()
        # backbone
        self.encoder = torch.hub.load(
            "facebookresearch/hiera",
            model=cfg.model,
            pretrained=cfg.pretrained,
            checkpoint=cfg.checkpoint,
        )
        # head
        self.encoder.head = nn.Identity()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """Forward pass of the Hiera model.

        Args:
            x (dict): Input dictionary containing:
                - imgs: Input video frames
                - gt_bboxes (optional): Ground truth bounding boxes for ROI Align

        Returns:
            torch.Tensor: Extracted features
        """
        if "gt_bboxes" in x:
            # Forward pass through Hiera backbone and get intermediate features
            _, intermediates = self.encoder(x["imgs"].squeeze(1), return_intermediates=True)

            # Temporal pooling of last stage
            feats = intermediates[-1].mean(dim=1)  # N, T, H, W, C --> N, H, W, C
            feats = feats.permute(0, 3, 1, 2)  # N, H, W, C --> N, C, H, W
            spatial_scale = feats.shape[-1] / 224

            # Prepare bounding boxes for ROI Align
            bboxes = x["gt_bboxes"]
            batch_indices = (
                torch.arange(bboxes.shape[0]).unsqueeze(1).to(bboxes.device).to(bboxes.dtype)
            )
            bboxes = torch.cat((batch_indices, bboxes), dim=1)
            bboxes = bboxes.to(feats.dtype)

            # Apply ROI Align to extract region-specific features
            res = roi_align(
                feats,
                bboxes,
                output_size=(feats.shape[-2], feats.shape[-1]),
                spatial_scale=spatial_scale,
                sampling_ratio=-1,
                aligned=True,
            )
            # Spatial pooling
            res = self.avg_pool(res).squeeze()
        else:
            # Global feature extraction without ROI Align
            res = self.encoder(x["imgs"].squeeze(1))

        return res
