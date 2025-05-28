#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import os
import shutil
from glob import glob

import joblib
from tqdm import tqdm


class FeatureExtractionManager:
    def __init__(self, config):
        self.config = config
        self.ext_name = config.get("extraction_name", "pred")

        self.output_path = config.output_path
        os.makedirs(self.output_path, exist_ok=True)

        self.dump_folder = os.path.join(self.output_path, "dumps")
        os.makedirs(self.dump_folder, exist_ok=True)

    def clean(self):
        if os.path.exists(self.dump_folder):
            shutil.rmtree(self.dump_folder)

    def save_extraction(self, extracted_data):
        output_path = os.path.join(self.output_path, f"{self.ext_name}.pkl")
        joblib.dump(extracted_data, output_path)
        print(f"Extraction data saved to {output_path}")

    def dump(self, batch_idx, batch, output):
        """
        Dump predictions and features for a batch.

        Args:
            batch_idx (int): Index of the batch.
            batch (dict): Contains batch data, including 'imgs' and 'instance_name'.
            output (torch.Tensor): Model output corresponding to the batch.
        """
        batch_size = next(v.shape[0] for v in output.values() if v is not None)
        instance_names = batch["instance_name"]
        feature_instances = {
            instance_names[i]: {key: output[key][i].cpu().numpy() for key in output}
            for i in range(batch_size)
        }
        dump_path = os.path.join(self.dump_folder, f"{batch_idx}.pkl")
        joblib.dump(feature_instances, dump_path)

    @staticmethod
    def decode_instance_name(instance_name):
        """
        Decode the components of an instance name.

        Args:
            instance_name (str): The full instance name.

        Returns:
            tuple: Decoded components (clip_name, key_frame, pid, hand).
        """
        parts = instance_name.split("/")
        clip_name = parts[0]
        key_frame = int(parts[1])
        pid = int(parts[2])
        hand = parts[3] if len(parts) > 3 else "body"
        return clip_name, key_frame, pid, hand

    def collect(self):
        """
        Collect all dumped features and predictions, organize them, and save them to a single file.
        """
        extracted_data = {}
        all_pkl_files = glob(os.path.join(self.dump_folder, "*.pkl"))

        for pkl_file in tqdm(all_pkl_files, desc="Collecting features"):
            batch_data = joblib.load(pkl_file)

            for instance_name, features in batch_data.items():
                clip_name, key_frame, pid, hand = self.decode_instance_name(instance_name)

                extracted_data.setdefault(clip_name, {}).setdefault(pid, {}).setdefault(
                    key_frame, {}
                )[hand] = features

        self.save_extraction(extracted_data)
        self.clean()
