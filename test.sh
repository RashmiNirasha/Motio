#!/bin/bash

#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

# Set the model, input type, and task 
#------------------------------------------------------
MODEL="hiera" # choices: {poseconv3d, rgbposeconv3d, hiera}
INPUT_TYPE="hand" # choices: {body, hand}
TASK="manipulation" # choices: {object, manipulation}

PATH_TO_CKPTS="./data/childplay_hand/checkpoints"
CKPT_PATH="${PATH_TO_CKPTS}/${MODEL}_${TASK}_${INPUT_TYPE}.ckpt"

#------------------------------------------------------
python ./test.py --config-name "${MODEL}_${TASK}.yaml" ckpt_path="$CKPT_PATH" \
    configs.task="$TASK" configs.input_type="$INPUT_TYPE"