#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#
from decord import VideoReader, cpu

from src.datamodules.components.childplay_hand import ChildPlayHandDataset


class Video_Dataset(ChildPlayHandDataset):
    def __init__(self, args, split, transform_pipeline, video_path, path_to_pose):
        super().__init__(args, split, transform_pipeline, path_to_pose)
        self.path_to_pose = path_to_pose
        self.video_name = "video"
        self.video_reader = VideoReader(video_path, ctx=cpu(0))

        self.load_pose_data()
        self.load_annotation()

    def on_init(self):
        """initialization hook."""
        pass

    def load_annotation(self):
        """Reads all the csv files of {split} and genrates the sample keys."""
        self.keys = self.generate_keys()

    def generate_keys(self):
        """Generate keys."""
        keys = []
        for pid in self.pose_data[self.video_name].keys():
            frame_indices = list(
                range(0, self.pose_data[self.video_name][pid]["body_keypoint"].shape[1])
            )
            for frame_idx in frame_indices:
                if self.args.input_type == "body":
                    keys.append((self.video_name, frame_idx, pid))
                elif self.args.input_type == "hand":
                    keys.append((self.video_name, frame_idx, pid, "left"))
                    keys.append((self.video_name, frame_idx, pid, "right"))
                else:
                    raise NotImplementedError(f"Invalid input_type: {self.args.input_type}")
        self.clip_to_frame_numbs = {self.video_name: frame_indices}

        return keys

    def add_labels(self, sample):
        """Add labels to the sample based on task type and input type."""
        return sample

    def add_extra_info(self, sample):
        sample["video_reader"] = self.video_reader
        return sample
