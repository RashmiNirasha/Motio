#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import os
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


def track_people(video_path, output_path, yolo_model="yolo11m"):
    print("Tracking started ...")
    save_file = os.path.join(output_path, "track.csv")

    model = YOLO(f"{yolo_model}.pt")

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Failed to open video: {video_path}"

    # Prepare tracking
    results = []
    frame_id = 0

    # Run tracker
    for track in model.track(source=video_path, stream=True, tracker="botsort.yaml", persist=True):
        frame_id += 1

        # If no tracking IDs, fill with -1
        if track.boxes.id is None:
            results.append(
                {
                    "clip": Path(video_path).name,
                    "frame": frame_id,
                    "pid": -1,
                    "p_xmin": -1,
                    "p_ymin": -1,
                    "p_xmax": -1,
                    "p_ymax": -1,
                }
            )
            continue

        ids = track.boxes.id.cpu().numpy().astype(int)
        boxes = track.boxes.xyxy.cpu().numpy().astype(int)
        classes = track.boxes.cls.cpu().numpy().astype(int)

        # Filter only 'person' class (class ID 0)
        for pid, (xmin, ymin, xmax, ymax), cls in zip(ids, boxes, classes):
            if cls != 0:
                continue

            results.append(
                {
                    "clip": Path(video_path).name,
                    "frame": frame_id,
                    "pid": pid,
                    "p_xmin": xmin,
                    "p_ymin": ymin,
                    "p_xmax": xmax,
                    "p_ymax": ymax,
                }
            )

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(save_file, index=False)
    print("Tracking Done!")

    return save_file
