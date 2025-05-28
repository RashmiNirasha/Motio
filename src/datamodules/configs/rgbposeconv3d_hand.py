#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

# https://github.com/kennymckormick/pyskl/blob/main/configs/rgbpose_conv3d/rgbpose_conv3d.py
left_kp = list(range(0, 21))
right_kp = list(range(0, 21))
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type="ClipSampler"),
    dict(type="MMDecode"),
    dict(type="MMCompact_BBox", hw_ratio=1.0, allow_imgpad=False, padding=0),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type="GeneratePoseTarget",
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(
        type="Collect",
        keys=["imgs", "heatmap_imgs", "label", "img_norm_cfg", "instance_name"],
        meta_keys=[],
    ),
    dict(type="ToTensor", keys=["imgs", "heatmap_imgs", "label"]),
]

val_pipeline = [
    dict(type="ClipSampler"),
    dict(type="MMDecode"),
    dict(type="MMCompact_BBox", hw_ratio=1.0, allow_imgpad=False, padding=0),
    dict(type="Resize", scale=(256, 256), keep_ratio=False),
    dict(
        type="GeneratePoseTarget",
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(
        type="Collect",
        keys=["imgs", "heatmap_imgs", "label", "img_norm_cfg", "instance_name"],
        meta_keys=[],
    ),
    dict(type="ToTensor", keys=["imgs", "heatmap_imgs", "label"]),
]
test_pipeline = [
    dict(type="ClipSampler"),
    dict(type="MMDecode"),
    dict(type="MMCompact_BBox", hw_ratio=1.0, allow_imgpad=False, padding=0),
    dict(type="Resize", scale=(256, 256), keep_ratio=False),
    dict(
        type="GeneratePoseTarget",
        sigma=0.7,
        use_score=True,
        with_kp=True,
        with_limb=False,
        scaling=0.25,
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(
        type="Collect",
        keys=["imgs", "heatmap_imgs", "label", "img_norm_cfg", "instance_name"],
        meta_keys=[],
    ),
    dict(type="ToTensor", keys=["imgs", "heatmap_imgs", "label"]),
]


pipelines = {
    "train": train_pipeline,
    "val": val_pipeline,
    "test": test_pipeline,
}
