#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
img_norm_cfg = dict(
    mean=[0.45 * 255.0, 0.45 * 255.0, 0.45 * 255.0],
    std=[0.225 * 255.0, 0.225 * 255.0, 0.255 * 255.0],
    to_bgr=False,
)

train_pipeline = [
    dict(type="ClipSampler"),
    dict(type="RGBDecode"),
    dict(type="MMCompact_BBox", hw_ratio=1.0, allow_imgpad=False, padding=0.0),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(
        type="Collect",
        keys=["imgs", "gt_bboxes", "label", "img_norm_cfg", "instance_name"],
        meta_keys=[],
    ),
    dict(type="ToTensor", keys=["imgs", "gt_bboxes", "label"]),
]

val_pipeline = [
    dict(type="ClipSampler"),
    dict(type="RGBDecode"),
    dict(type="MMCompact_BBox", hw_ratio=1.0, allow_imgpad=False, padding=0.0),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(
        type="Collect",
        keys=["imgs", "gt_bboxes", "label", "img_norm_cfg", "instance_name"],
        meta_keys=[],
    ),
    dict(type="ToTensor", keys=["imgs", "gt_bboxes", "label"]),
]
test_pipeline = [
    dict(type="ClipSampler"),
    dict(type="RGBDecode"),
    dict(type="MMCompact_BBox", hw_ratio=1.0, allow_imgpad=False, padding=0.0),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(
        type="Collect",
        keys=["imgs", "gt_bboxes", "label", "img_norm_cfg", "instance_name"],
        meta_keys=[],
    ),
    dict(type="ToTensor", keys=["imgs", "gt_bboxes", "label"]),
]

pipelines = {
    "train": train_pipeline,
    "val": val_pipeline,
    "test": test_pipeline,
}
