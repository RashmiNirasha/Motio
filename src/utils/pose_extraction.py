#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import argparse
import os
import shutil
import warnings
from glob import glob

import joblib
import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from tqdm import tqdm

from mmpose.apis import inference_topdown, init_model
from src.datamodules.components.transforms import get_frame_path
from src.utils.generate_hand_bbox import HandBBoxGenerator

warnings.filterwarnings("ignore")

CONFIG_FILES = {
    "body": "src/utils/configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py",
    "whole_body": "src/utils/configs/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py",
}

CHECKPOINTS = {
    "body": "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth",
    "whole_body": "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth",
}

keypoint_ids = {
    "left": list(range(91, 112)),
    "right": list(range(112, 133)),
}


class UnifiedMMPoseExtractor:
    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.body_num_keypoints = 17
        self.hand_num_keypoints = 21
        # Initialize models for body and hand
        self.body_model = init_model(CONFIG_FILES["body"], CHECKPOINTS["body"], device=device)
        self.hand_model = init_model(
            CONFIG_FILES["whole_body"], CHECKPOINTS["whole_body"], device=device
        )
        self.hand_bbox_generator = HandBBoxGenerator(extend_factor=0.5, ratio_body=0.4)

    @staticmethod
    def init_data(num_frames):
        return {
            "img_shape": None,
            "original_shape": None,
            "body_bbox": np.full((1, num_frames, 4), -1.0, dtype=np.float32),
            "body_keypoint": np.zeros((1, num_frames, 17, 2), dtype=np.float32),
            "body_keypoint_score": np.zeros((1, num_frames, 17), dtype=np.float32),
            "left_hand_keypoint": np.zeros((1, num_frames, 21, 2), dtype=np.float32),
            "right_hand_keypoint": np.zeros((1, num_frames, 21, 2), dtype=np.float32),
            "left_hand_keypoint_score": np.zeros((1, num_frames, 21), dtype=np.float32),
            "right_hand_keypoint_score": np.zeros((1, num_frames, 21), dtype=np.float32),
            "left_hand_bbox": np.full((1, num_frames, 4), -1.0, dtype=np.float32),
            "right_hand_bbox": np.full((1, num_frames, 4), -1.0, dtype=np.float32),
        }

    def update_extraction(self, res_body, res_hand, tracks, pid, fid):
        tracks[pid]["img_shape"] = res_body[0].img_shape
        tracks[pid]["original_shape"] = res_body[0].img_shape

        # Body keypoints and scores
        body_pose = res_body[0].pred_instances
        tracks[pid]["body_keypoint"][0, fid] = body_pose.keypoints[0]
        tracks[pid]["body_keypoint_score"][0, fid] = body_pose.keypoint_scores[0]

        # Hand keypoints and scores
        hand_pose = res_hand[0].pred_instances
        tracks[pid]["left_hand_keypoint"][0, fid] = hand_pose.keypoints[0][keypoint_ids["left"]]
        tracks[pid]["right_hand_keypoint"][0, fid] = hand_pose.keypoints[0][keypoint_ids["right"]]
        tracks[pid]["left_hand_keypoint_score"][0, fid] = hand_pose.keypoint_scores[0][
            keypoint_ids["left"]
        ]
        tracks[pid]["right_hand_keypoint_score"][0, fid] = hand_pose.keypoint_scores[0][
            keypoint_ids["right"]
        ]

        # Hand bounding boxes
        tracks[pid]["left_hand_bbox"][0, fid] = self.hand_bbox_generator.compute_hand_bbox(
            tracks[pid]["img_shape"],
            tracks[pid]["body_bbox"][0, fid],
            tracks[pid]["body_keypoint"][0, fid],
            elbow_idx=7,
            wrist_idx=9,
        )
        tracks[pid]["right_hand_bbox"][0, fid] = self.hand_bbox_generator.compute_hand_bbox(
            tracks[pid]["img_shape"],
            tracks[pid]["body_bbox"][0, fid],
            tracks[pid]["body_keypoint"][0, fid],
            elbow_idx=8,
            wrist_idx=10,
        )

    def __call__(self, tracks, frame_paths=None, video_path=None):
        frame_ids = sorted({f for person in tracks.values() for f in person})
        n_frames = len(frame_ids)

        extractions = {pid: self.init_data(n_frames) for pid in tracks}

        if frame_paths is None:
            vr = VideoReader(str(video_path), ctx=cpu(0))

        for frame_idx, frame_id in tqdm(enumerate(frame_ids), total=n_frames):
            frame = frame_paths[frame_id] if frame_paths else vr[frame_idx].asnumpy()

            for pid, person_track in tracks.items():
                if frame_id not in person_track:
                    continue

                bbox = person_track[frame_id]
                extractions[pid]["body_bbox"][0, frame_idx] = bbox
                bbox_arr = np.array([bbox])

                res_body = inference_topdown(
                    self.body_model, frame, bboxes=bbox_arr, bbox_format="xyxy"
                )
                res_hand = inference_topdown(
                    self.hand_model, frame, bboxes=bbox_arr, bbox_format="xyxy"
                )

                self.update_extraction(res_body, res_hand, extractions, pid, frame_idx)

        return extractions


######################################################
# Utils
######################################################
def dump_extraction(extraction, save_file):
    joblib.dump(extraction, save_file)


def clean(dump_folder):
    """
    Cleans up by removing the dump folder. To be called after all processing is done.
    """
    if os.path.exists(dump_folder):
        shutil.rmtree(dump_folder)


def collect_extractions(dump_folder, save_file):
    """
    Collect all per-video dumped pkl files and unfies them into one
    """
    extracted_data = {}
    all_pkl_files = glob(os.path.join(dump_folder, "*.pkl"))

    for pkl_file in tqdm(all_pkl_files, desc="Collecting pkls"):
        video_data = joblib.load(pkl_file)
        extracted_data.update(video_data)

    dump_extraction(extracted_data, save_file)
    clean(dump_folder)


def validate_csv(path):
    required_cols = {"clip", "frame", "pid", "p_xmin", "p_ymin", "p_xmax", "p_ymax"}
    df = pd.read_csv(path)
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV at {path} is missing required columns: {required_cols - set(df.columns)}"
        )


def read_tracks(path_to_csv, root_video, dataset="childplay"):
    """
    Loads all tracks of a video from a CSV annotation file.
    """
    # Validate input
    assert path_to_csv.endswith(".csv"), "File must be a CSV."
    assert os.path.exists(path_to_csv), "File does not exist."

    validate_csv(path_to_csv)

    # Load data
    df = pd.read_csv(path_to_csv)
    required_columns = ["frame", "pid", "p_xmin", "p_ymin", "p_xmax", "p_ymax"]
    assert all(
        col in df.columns for col in required_columns
    ), f"CSV must contain columns: {required_columns}"

    # Prepare track dictionary
    track_dict = {}
    frame_numbers = sorted(df["frame"].unique())

    frame_paths = {}
    clip = path_to_csv.split("/")[-1][:-4]

    for pid in df["pid"].unique():
        track_dict[pid] = {}
        person_data = df[df["pid"] == pid]

        for frame_id in frame_numbers:
            frame_data = person_data[person_data["frame"] == frame_id]
            if not frame_data.empty:
                track_dict[pid][frame_id] = frame_data[
                    ["p_xmin", "p_ymin", "p_xmax", "p_ymax"]
                ].iloc[0]

            if frame_id not in frame_paths and dataset == "childplay":
                frame_paths[frame_id] = get_frame_path(root_video, clip, frame_id)

    return track_dict, frame_paths


######################################################
# Extraction
######################################################
def run_extraction(cfg):
    """Pose Extraction."""
    list_dir_csv = glob(
        os.path.join(os.path.join(cfg.root_data, "annotations", cfg.split), "*.csv")
    )
    # Create dirs
    root_pose = os.path.join(cfg.root_data, "pose")
    os.makedirs(root_pose, exist_ok=True)

    tmp_folder = os.path.join(root_pose, f"pose_{cfg.split}_tmp")
    os.makedirs(tmp_folder, exist_ok=True)

    extractor = UnifiedMMPoseExtractor(cfg.device)

    for path_to_csv in tqdm(list_dir_csv):
        # Video info
        video_name = path_to_csv.split("/")[-1][:-4]
        # Run Extraction
        tracks, frame_paths = read_tracks(path_to_csv, cfg.root_video, dataset=cfg.dataset)
        extractions = extractor(tracks, frame_paths=frame_paths)
        # Dump
        dump_extraction({video_name: extractions}, save_file=f"{tmp_folder}/{video_name}.pkl")

    collect_extractions(tmp_folder, save_file=f"{root_pose}/cp_pose_{cfg.split}.pkl")
    print("Extraction Done!!!")


######################################################
# Extraction - Demo Version
######################################################


def run_pose_extraction_demo(video_path, path_to_tracks, output_path):
    """Extract poses from the input video and save them as a pickle file."""
    output_file = os.path.join(output_path, "pose.pkl")

    tracks, _ = read_tracks(path_to_tracks, output_path, dataset="demo")
    extractor = UnifiedMMPoseExtractor()
    extractions = extractor(tracks, video_path=video_path)

    dump_extraction({"video": extractions}, save_file=output_file)
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration for extraction.")
    # Add arguments
    parser.add_argument("--dataset", type=str, default="childplay")
    parser.add_argument("--root_data", type=str, default="./data/childplay_hand/data")
    parser.add_argument("--root_video", type=str, default="./data/ChildPlay-gaze")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--split", type=str, default=None, choices=["train", "test", "val"])

    cfg = parser.parse_args()

    run_extraction(cfg)
