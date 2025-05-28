#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import os

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from decord import VideoReader, cpu

from src.evaluation.cp_hand_eval import TASKS
from src.utils.configs.utils import (
    hand_name_to_id,
    left_hand_ids,
    right_hand_ids,
    skeleton_info,
)


# CONSTANTS
def generate_bgr_colors(n):
    cmap = plt.cm.get_cmap("tab20", n)  # or use 'hsv' for more diversity
    colors = [tuple(int(255 * c) for c in cmap(i)[:3][::-1]) for i in range(n)]
    return colors


COLORS_BGR = generate_bgr_colors(10)

HAND_KEYPOINTS = {"left": left_hand_ids, "right": right_hand_ids}

SKELETON_MAP = {
    "coco": [
        (0, 1, "f"),
        (0, 2, "f"),
        (1, 3, "f"),
        (2, 4, "f"),
        (0, 5, "t"),
        (0, 6, "t"),
        (5, 7, "ru"),
        (6, 8, "lu"),
        (7, 9, "ru"),
        (8, 10, "lu"),
        (5, 11, "t"),
        (6, 12, "t"),
        (11, 13, "ld"),
        (12, 14, "rd"),
        (13, 15, "ld"),
        (14, 16, "rd"),
    ]
}


def logit_to_label(pred):
    """Convert logits to class labels using sigmoid or softmax."""
    pred_tensor = torch.tensor(pred, dtype=torch.float32)
    return (
        (pred_tensor.sigmoid() > 0.5).int().item()
        if pred_tensor.numel() == 1
        else pred_tensor.argmax().item()
    )


def draw_keypoints(frame, kp, kpscore, color, layout="coco", thre=0.2, thick=2):
    """Draw body keypoints."""
    # Based on: https://github.com/kennymckormick/pyskl/blob/main/pyskl/utils/visualize.py
    edges = SKELETON_MAP[layout]
    kp = np.concatenate([kp, kpscore[..., None]], -1)
    ske = kp[0]

    for e in edges:
        st, ed, co = e
        j1, j2 = ske[st], ske[ed]
        j1x, j1y, j2x, j2y = int(j1[0]), int(j1[1]), int(j2[0]), int(j2[1])
        conf = min(j1[2], j2[2])
        if conf > thre:
            cv2.line(frame, (j1x, j1y), (j2x, j2y), color, thick)


def draw_hand_keypoints(frame, kp, kpscore, hand, color, thre=0.2, thick=2):
    """Draw hand keypoints."""
    kp = np.concatenate([kp, kpscore[..., None]], -1)
    ske = kp[0]

    for k_sinfo, v_sinfo in skeleton_info.items():
        link = v_sinfo["link"]
        sk1_id = hand_name_to_id[link[0]]
        sk2_id = hand_name_to_id[link[1]]
        if sk1_id not in HAND_KEYPOINTS[hand] or sk2_id not in HAND_KEYPOINTS[hand]:
            continue
        sk1_id = HAND_KEYPOINTS[hand].index(sk1_id)
        sk2_id = HAND_KEYPOINTS[hand].index(sk2_id)

        j1, j2 = ske[sk1_id], ske[sk2_id]
        j1x, j1y, j2x, j2y = int(j1[0]), int(j1[1]), int(j2[0]), int(j2[1])
        conf = min(j1[2], j2[2])
        if conf > thre:
            cv2.line(frame, (j1x, j1y), (j2x, j2y), color, thick)

    for j_id in range(ske.shape[0]):
        if j_id not in HAND_KEYPOINTS[hand]:
            continue
        x, y, conf = ske[j_id]
        if conf > thre:
            cv2.circle(frame, (x, y), radius=1, color=color, thickness=thick)


def draw_bbox(frame, bbox, color, thickness=2):
    """Draw bounding box."""
    bbox = tuple(map(int, bbox))
    cv2.rectangle(frame, bbox[:2], bbox[2:], color, thickness)


def annotate_text(frame, coords, text, text_color=(0, 255, 0), bg_color=(255, 255, 255)):
    """Annotate frame with a label."""
    x, y = map(int, coords[:2])
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    cv2.rectangle(frame, (x, y - text_size[1]), (x + text_size[0], y), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)


def draw_annotations(cfg, frame, person_data, predictions, frame_id, color):
    """Draw all relevant annotations on a video frame."""

    for hand in ["left", "right"]:
        if cfg.draw_hand_action:
            label_idx = logit_to_label(predictions[hand][f"pred_{cfg.task}"])
            label_text = TASKS[cfg.task]["idx_to_class"][label_idx]
            if label_text != "background":
                wrist_id = 9 if hand == "left" else 10
                annotate_text(
                    frame,
                    person_data["body_keypoint"][0, frame_id, wrist_id],
                    label_text,
                    text_color=color,
                )
                # Bounding Box
                draw_bbox(frame, person_data[f"{hand}_hand_bbox"][0, frame_id], color)

        # Hand keypoints
        if cfg.draw_hand_skeletons:
            draw_hand_keypoints(
                frame,
                person_data[f"{hand}_hand_keypoint"][:, frame_id],
                person_data[f"{hand}_hand_keypoint_score"][:, frame_id],
                hand,
                color,
            )

    # Body keypoints
    if cfg.draw_body_skeletons:
        draw_keypoints(
            frame,
            person_data["body_keypoint"][:, frame_id],
            person_data["body_keypoint_score"][:, frame_id],
            color,
        )


# Main Drawing Func
def draw_on_video(cfg, video_path, path_to_pose, path_to_action, output_path, fps=30):
    """Load video, overlay annotations, and save output."""
    # Load data
    pose_data = joblib.load(path_to_pose)["video"]
    predictions = joblib.load(path_to_action)["video"]

    video_reader = VideoReader(video_path, ctx=cpu(0))

    # Setup video writer
    height, width = video_reader[0].shape[:2]
    writer = cv2.VideoWriter(
        os.path.join(output_path, "demo_video.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # Process frames
    for frame_id, frame in enumerate(video_reader):
        frame = frame.asnumpy()
        for person_id, person_data in pose_data.items():
            draw_annotations(
                cfg,
                frame,
                person_data,
                predictions[person_id][frame_id],
                frame_id,
                COLORS_BGR[person_id % len(COLORS_BGR)],
            )
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Video saved at {output_path}")
