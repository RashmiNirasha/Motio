#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import importlib

PIPELINE_NAMES = [
    "hiera_body", "hiera_hand",
    "mstcn_body", "mstcn_hand",
    "poseconv3d_body", "poseconv3d_hand",
    "rgbposeconv3d_body", "rgbposeconv3d_hand",
    "sstcn_body", "sstcn_hand",
]

PIPELINE_MAP = {
    name: importlib.import_module(f"src.datamodules.configs.{name}")
    for name in PIPELINE_NAMES
}

def generate_data_pipeline(config, split):
    module = PIPELINE_MAP.get(config.pipeline)
    if module is None:
        raise NotImplementedError(f"Invalid pipeline: {config.pipeline}")
    return module.pipelines[split]
