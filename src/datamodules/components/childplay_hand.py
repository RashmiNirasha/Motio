#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import os
from glob import glob

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.datamodules.components.transforms import Compose
from src.datamodules.components.utils import generate_keys


class ChildPlayHandDataset(Dataset):
    """ChildPlay-Hand Dataset for Recognition."""

    def __init__(self, args, split, transform_pipeline, path_to_pose=None):
        super().__init__()

        self.args = args
        self.split = split
        self.tasks = args.task.split("_")
        self.path_to_pose = path_to_pose

        assert split in ["train", "val", "test"], "Invalid dataset split!"
        assert args.input_type in ["body", "hand"], "Invalid input type!"

        self.on_init()
        self.transforms = Compose(transform_pipeline)

    def __len__(self):
        return len(self.keys)

    def on_init(self):
        """initialization hook."""
        frames_path = os.path.join(self.args.root_vid, "images")
        assert os.path.exists(frames_path), (
            f"Missing extracted frames at: {frames_path}. "
            "Follow the instructions in the original ChildPlay dataset to extract frames."
        )
        assert os.path.exists(self.args.root_data), (
            f"ChildPlay-Hand annotations not found at: {self.args.root_data}. "
            "Refer to data/README.md for download and setup instructions."
        )
        self.load_pose_data()
        self.load_annotation()

    def load_annotation(self):
        """Reads CSV annotation files for the given split and generates sample keys."""
        annotation_path = os.path.join(self.args.root_data, "annotations", self.split)
        annotation_files = glob(os.path.join(annotation_path, "*.csv"))

        li = []
        self.clip_to_frame_numbs = {}

        for file in annotation_files:
            df = pd.read_csv(file)
            li.append(df)
            clip_name = os.path.basename(file)[:-4]
            self.clip_to_frame_numbs[clip_name] = sorted(set(df["frame"]))

        annotations = pd.concat(li, axis=0, ignore_index=True)

        self.groups = ["clip", "frame", "pid"] + (
            ["hand"] if self.args.input_type == "hand" else []
        )
        self.annotations, self.keys = self.generate_keys(annotations)

    def generate_keys(self, annotations):
        """Generate keys for indexing annotations."""
        if self.split == "train" and not self.args.extraction:
            annotations = annotations.groupby(self.groups)
            keys = generate_keys(self.args, key_stride=3)
        else:
            if self.args.exclude_person_not_visible_test:
                annotations = annotations[annotations["p_visible"] != 0].reset_index(drop=True)

            annotations = annotations.groupby(self.groups)
            keys = np.array(list(annotations.groups.keys()))

        return annotations, keys

    def load_pose_data(self):
        """Load precomputed pose data for the given split."""
        if self.path_to_pose is None:
            self.path_to_pose = f"{self.args.root_data}/pose/cp_pose_{self.split}.pkl"
        self.pose_data = joblib.load(self.path_to_pose)

    def add_labels(self, sample):
        """Add classification labels to the sample based on task and input type."""
        sample_annotations = self.annotations.get_group(sample["key"])

        if self.args.input_type == "hand":
            for task in self.tasks:
                sample[f"{task}_task"] = torch.tensor(
                    sample_annotations[f"{task}_task"].values[0], dtype=torch.long
                )
        else:
            self._add_hand_labels(sample, sample_annotations)

        return sample

    def _add_hand_labels(self, sample, sample_annotations):
        """Add hand-specific labels, including flipped labels for augmentation."""
        sample["label_flip"] = {}

        for task in self.tasks:
            for hand in ["left", "right"]:
                key = f"{task}_task"
                label = sample_annotations[sample_annotations["hand"] == hand][key].values[0]
                sample[f"{hand}_{task}_task"] = torch.tensor(label, dtype=torch.long)

                # Flipped labels for augmentation
                opposite_hand = "right" if hand == "left" else "left"
                flipped_label = sample_annotations[sample_annotations["hand"] == opposite_hand][
                    key
                ].values[0]
                sample["label_flip"][f"{hand}_{task}_task"] = torch.tensor(
                    flipped_label, dtype=torch.long
                )

    def _add_pose_data(self, sample):
        """Add pose-related data, including keypoints and bounding boxes."""
        sample["Pose"] = {
            "sampling_rate": 1,
            "window_size": self.args.window_size,
            "clip_len": self.args.window_size,
        }

        pose_data = self.pose_data[sample["clip"]][sample["pid"]]
        prefix = f"{sample['key'][-1]}_hand" if self.args.input_type == "hand" else "body"

        sample["n_keypoints"] = 21 if self.args.input_type == "hand" else 17
        sample["keypoint"] = pose_data[f"{prefix}_keypoint"]
        sample["keypoint_score"] = pose_data[f"{prefix}_keypoint_score"]
        sample["bbox"] = pose_data[f"{prefix}_bbox"]
        sample["img_shape"] = pose_data["img_shape"]

        return sample

    def _add_rgb_data(self, sample):
        """Add RGB-related data and bounding boxes."""
        sample["RGB"] = {
            "sampling_rate": self.args.sampling_rate,
            "window_size": self.args.window_size,
            "clip_len": self.args.window_size // self.args.sampling_rate,
        }

        pose_data = self.pose_data[sample["clip"]][sample["pid"]]
        prefix = f"{sample['key'][-1]}_hand" if self.args.input_type == "hand" else "body"

        sample["bbox"] = pose_data[f"{prefix}_bbox"]
        sample["img_shape"] = pose_data["img_shape"]

        return sample

    def add_data(self, sample):
        """Add general metadata and modality-specific data to the sample."""
        sample.update(
            {
                "window_size": self.args.window_size,
                "clip_len": self.args.window_size,
                "modality": self.args.modality,
                "root_vid": self.args.root_vid,
                "input_type": self.args.input_type,
                "num_clips": 1,
                "instance_name": "/".join(map(str, sample["key"])),
                "frame_numbs": self.clip_to_frame_numbs[sample["clip"]],
            }
        )

        if "Pose" in self.args.modality:
            self._add_pose_data(sample)
        if "RGB" in self.args.modality:
            self._add_rgb_data(sample)

        return sample

    def add_extra_info(self, sample):
        """Hook to add extra information to a sample. Currently unused."""
        return sample

    @staticmethod
    def _check_key(key):
        """Ensure the key format is valid."""
        key = tuple(key)
        assert len(key) in [3, 4], f"Invalid key format: {key}"
        return (key[0], int(key[1]), int(key[2]), *key[3:])

    def __getitem__(self, index):
        """Retrieve and transform a sample based on its key."""
        key = self._check_key(self.keys[index])
        sample = {"key": key, "clip": key[0], "pid": key[2]}

        sample = self.add_labels(sample)
        sample = self.add_data(sample)
        sample = self.add_extra_info(sample)

        return self.transforms(sample)
