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

NUM_CLASS = {"object": 2, "manipulation": 5, "full": 10}


class SampleGenerator:
    def __init__(self, args, split):
        assert split in ["train", "val", "test"]
        self.args = args
        self.split = split
        os.makedirs(self.args.storage_folder, exist_ok=True)

        self.interaction_key = f"{args.task}_task"
        self.index_s = 0

        # Batch Size
        if split in ["val", "test"]:
            self.org_batch_size = 1
        else:
            self.org_batch_size = args.org_batch_size

        # Load annotations
        self.annotations, self.keys = self.load_annotation()
        self.reset()

        # Load Pose or Features
        self.feat_data = self.load_feat_data()

    def __len__(self):
        return len(self.keys) // self.org_batch_size

    def load_feat_data(self):
        return joblib.load(
            f"{self.args.root_data}/feats/childplay_{self.args.feat_name}_{self.args.input_type}_{self.args.task}_{self.split}.pkl"
        )

    def reset(self):
        self.index_s = 0
        if self.split == "train":
            np.random.shuffle(self.keys)

    def generate_keys(self, annotations):
        annotations = annotations.groupby(["clip", "pid", "hand"])
        keys = np.array(list(annotations.groups.keys()))

        return annotations, keys

    @staticmethod
    def _get_len_frames_person(df):
        return df.groupby("pid")["frame"].nunique().to_dict()

    def load_annotation(self):
        self.cached_len_instance = {}
        annotation_files = glob(
            os.path.join(os.path.join(self.args.root_data, "annotations", self.split), "*.csv")
        )
        li = []
        for file in annotation_files:
            df = pd.read_csv(file).copy()
            li.append(df)
            clip_name = file.split("/")[-1][:-4]
            self.cached_len_instance[clip_name] = self._get_len_frames_person(df)
        annotations = pd.concat(li, axis=0, ignore_index=True)
        # generate keys
        annotations, keys = self.generate_keys(annotations)

        return annotations, keys

    def get_len_instance(self, x):
        clip, pid, hand = x
        return self.cached_len_instance[clip][int(pid)]

    def get_feats_instance(self, clip, pid, hand):
        return self.feat_data[clip][pid][hand]["feats"]

    def get_labels_instance(self, clip, pid, hand):
        ins_annotations = self.annotations.get_group((clip, pid, hand))
        labels = ins_annotations[self.interaction_key].tolist()
        mask_person = ins_annotations["p_visible"].tolist()
        f_names = ins_annotations["frame"].tolist()

        return np.array(labels), np.array(mask_person), f_names

    def one_hand_data(self, sample, meta_sample, batch, max_len_seq_bacth):
        # Feature grid
        input_feats = torch.zeros(
            len(batch), self.args.feat_dim, max_len_seq_bacth, dtype=torch.float
        )  # B, D, T
        # Labels
        labels = torch.ones(len(batch), max_len_seq_bacth, dtype=torch.long) * (-100)  # B, T
        # Mask
        mask = torch.zeros(
            len(batch), NUM_CLASS[self.args.task], max_len_seq_bacth, dtype=torch.float
        )  # B, C, T
        # Instance names
        instance_names = []
        frame_names = []

        for i in range(len(batch)):
            clip, pid, hand = batch[i]
            pid = int(pid)

            feat_ins = self.get_feats_instance(clip, pid, hand)  # T x D
            label_ins, mask_person, f_names = self.get_labels_instance(clip, pid, hand)  # T
            T_feat = feat_ins.shape[0]

            input_feats[i, :, :T_feat] = torch.from_numpy(feat_ins).permute(1, 0)
            labels[i, :T_feat] = torch.from_numpy(label_ins)
            mask[i, :, :T_feat] = torch.from_numpy(mask_person).float()[None, :] * torch.ones(
                NUM_CLASS[self.args.task], T_feat
            )

            instance_names.append(f"{clip}/{pid}/{hand}")
            frame_names.append(f_names)

        # fill in sample
        sample["feats"] = input_feats
        sample["labels"] = labels
        sample["mask"] = mask
        meta_sample["instance_name"] = instance_names
        meta_sample["frame_names"] = frame_names

        return sample, meta_sample

    def sample_instance(self, sample, meta_sample):
        batch = self.keys[self.index_s : self.index_s + self.org_batch_size]
        self.index_s += self.org_batch_size

        if self.split == "train":
            np.random.shuffle(batch)
        max_len_seq_batch = max(list(map(self.get_len_instance, batch)))

        sample, meta_sample = self.one_hand_data(sample, meta_sample, batch, max_len_seq_batch)

        return sample, meta_sample

    def __call__(self, sample, meta_sample):
        return self.sample_instance(sample, meta_sample)


# ----------------------------------------------------------------------------
#                     ChildPlay-Hand Dataset - Segmentation
# ----------------------------------------------------------------------------
class ChildPlayHandDataset_TAS(Dataset):
    def __init__(self, args, split, transform_pipeline=None):
        super().__init__()
        assert args.input_type in ["hand"]

        self.args = args
        self.split = split

        self.sample_generator = SampleGenerator(args, split)
        self.collect_keys = ["feats", "labels", "mask"]
        self.collect_meta_keys = ["instance_name", "frame_names"]

    def __len__(self):
        return len(self.sample_generator)

    def reset(self):
        self.sample_generator.reset()

    def __getitem__(self, index):
        # Init
        sample = {"index": index}
        meta_sample = dict()
        # Generate
        sample, meta_sample = self.sample_generator(sample, meta_sample)
        # Collect
        sample = {k: v for k, v in sample.items() if k in self.collect_keys}
        meta_sample = {k: v for k, v in meta_sample.items() if k in self.collect_meta_keys}

        return sample, meta_sample
