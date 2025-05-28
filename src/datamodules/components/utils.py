#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import os
from glob import glob

import numpy as np
import pandas as pd


def load_annotations(root_dir, split):
    """
    Loads annotation CSV files from a given root directory and split.
    """
    annotation_files = glob(os.path.join(root_dir, "annotations", split, "*.csv"))
    return pd.concat((pd.read_csv(file) for file in annotation_files), ignore_index=True)


def filter_visible_annotations(annotations):
    """
    Filters out annotations where person is not visible.
    """
    return annotations[annotations["p_visible"] != 0].reset_index(drop=True)


def subsample_keys(annotations_grouped, key_stride):
    """
    Subsamples background keys while keeping all non-background keys.
    """
    group_is_bg = annotations_grouped["is_bg"].all().reset_index()
    bg_keys = group_is_bg[group_is_bg["is_bg"]][["clip", "frame", "pid"]].to_records(index=False)[
        ::key_stride
    ]
    non_bg_keys = group_is_bg[~group_is_bg["is_bg"]][["clip", "frame", "pid"]].to_records(
        index=False
    )
    return np.concatenate((bg_keys, non_bg_keys))


def generate_body_keys(annotations, key_stride, interaction_key):
    """
    Generates keys for body input.
    """
    annotations["is_bg"] = annotations[interaction_key] == 0
    grouped_annotations = annotations.groupby(["clip", "frame", "pid"])
    return subsample_keys(grouped_annotations, key_stride)


def generate_hand_keys(annotations, key_stride, interaction_key):
    """
    Generates keys for hand input.
    """
    annotations["is_bg"] = annotations[interaction_key] == 0
    grouped_annotations = annotations.groupby(["clip", "frame", "pid"])
    keys = subsample_keys(grouped_annotations, key_stride)
    keys_df = pd.DataFrame(keys.tolist(), columns=["clip", "frame", "pid"])
    filtered_annotations = annotations.merge(keys_df, on=["clip", "frame", "pid"], how="inner")
    return np.array(
        list(filtered_annotations.groupby(["clip", "frame", "pid", "hand"]).groups.keys())
    )


def generate_keys(args, key_stride=3):
    """
    Generates keys based on the input type (body or hand) with subsampling.
    """
    interaction_key = "manipulation_task"
    annotations = load_annotations(args.root_data, "train")
    annotations = filter_visible_annotations(annotations)

    if args.input_type == "hand":
        return generate_hand_keys(annotations, key_stride, interaction_key)
    elif args.input_type == "body":
        return generate_body_keys(annotations, key_stride, interaction_key)
    else:
        raise ValueError(f"Invalid input_type: {args.input_type}")
