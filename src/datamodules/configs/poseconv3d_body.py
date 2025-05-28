#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

# https://github.com/kennymckormick/pyskl/blob/main/configs/posec3d/slowonly_r50_ntu60_xsub/joint.py
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

train_pipeline = [
    dict(type="ClipSampler"),
    dict(type="PoseDecode"),
    dict(type="BBoxDecode"),
    dict(type="PoseCompact_BBox", hw_ratio=1.0, allow_imgpad=False, padding=0.0),
    dict(type="Resize", scale=(56, 56), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type="GeneratePoseTarget", with_kp=True, with_limb=False),
    dict(type="FormatShape", input_format="NCTHW_Heatmap"),
    dict(type="Collect", keys=["imgs", "label", "instance_name"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs", "label"]),
]

val_pipeline = [
    dict(type="ClipSampler"),
    dict(type="PoseDecode"),
    dict(type="BBoxDecode"),
    dict(type="PoseCompact_BBox", hw_ratio=1.0, allow_imgpad=False, padding=0.0),
    dict(type="Resize", scale=(64, 64), keep_ratio=False),
    dict(type="GeneratePoseTarget", with_kp=True, with_limb=False),
    dict(type="FormatShape", input_format="NCTHW_Heatmap"),
    dict(type="Collect", keys=["imgs", "label", "instance_name"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs", "label"]),
]

test_pipeline = [
    dict(type="ClipSampler"),
    dict(type="PoseDecode"),
    dict(type="BBoxDecode"),
    dict(type="PoseCompact_BBox", hw_ratio=1.0, allow_imgpad=False, padding=0.0),
    dict(type="Resize", scale=(64, 64), keep_ratio=False),
    dict(type="GeneratePoseTarget", with_kp=True, with_limb=False),
    dict(type="FormatShape", input_format="NCTHW_Heatmap"),
    dict(type="Collect", keys=["imgs", "label", "instance_name"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs", "label"]),
]

pipelines = {
    "train": train_pipeline,
    "val": val_pipeline,
    "test": test_pipeline,
}
