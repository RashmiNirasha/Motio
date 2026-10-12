import os
import cv2
import joblib
import numpy as np
import pandas as pd

PILOT_CSV = "infant_pilot_20.csv"
POSE_DIR = "data/childplay_hand/data/pose"
CLIP_DIR = "data/ChildPlay-gaze/clips"
OUTPUT_DIR = "pilot_grasp_videos"

os.makedirs(OUTPUT_DIR, exist_ok=True)

pilot = pd.read_csv(PILOT_CSV)

pose_cache = {}

connections = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (5,9),(9,10),(10,11),(11,12),     # middle
    (9,13),(13,14),(14,15),(15,16),   # ring
    (13,17),(17,18),(18,19),(19,20),  # little
    (0,17)
]


def get_pose(split):
    split = str(split).strip()

    if split not in pose_cache:
        path = os.path.join(
            POSE_DIR,
            f"cp_pose_{split}.pkl"
        )

        print("Loading pose file:", path)
        pose_cache[split] = joblib.load(path)

    return pose_cache[split]


index_rows = []

for event_num, (_, row) in enumerate(
    pilot.iterrows(),
    start=1
):

    clip = str(row["clip"])
    split = str(row["split"]).strip()
    pid = int(row["pid"])
    hand = str(row["hand"]).strip().lower()

    start_frame = int(row["start_frame"])
    end_frame = int(row["end_frame"])

    print("\n======================================")
    print(
        f"EVENT {event_num:02d}/20 | "
        f"{clip} | pid {pid} | {hand}"
    )
    print("======================================")

    video_path = os.path.join(
        CLIP_DIR,
        f"{clip}.mp4"
    )

    if not os.path.exists(video_path):
        print("❌ Clip video missing")
        continue

    pose = get_pose(split)

    if clip not in pose:
        print("❌ Clip missing from pose data")
        continue

    if pid not in pose[clip]:
        print("❌ Person ID missing from pose data")
        continue

    pdata = pose[clip][pid]

    bbox_key = f"{hand}_hand_bbox"
    kp_key = f"{hand}_hand_keypoint"
    score_key = f"{hand}_hand_keypoint_score"

    if bbox_key not in pdata:
        print("❌ Hand bbox missing")
        continue

    bbox_data = np.asarray(pdata[bbox_key])
    kp_data = np.asarray(pdata[kp_key])
    score_data = np.asarray(pdata[score_key])

    # ------------------------------------------
    # Stable crop covering the complete event
    # ------------------------------------------
    valid_boxes = []

    for frame in range(start_frame, end_frame + 1):

        box = np.asarray(
            bbox_data[0, frame - 1]
        ).astype(float)

        if (
            len(box) == 4
            and np.all(np.isfinite(box))
            and box[2] > box[0]
            and box[3] > box[1]
        ):
            valid_boxes.append(box)

    if not valid_boxes:
        print("❌ No valid hand boxes")
        continue

    valid_boxes = np.asarray(valid_boxes)

    x1 = valid_boxes[:,0].min()
    y1 = valid_boxes[:,1].min()
    x2 = valid_boxes[:,2].max()
    y2 = valid_boxes[:,3].max()

    bw = x2 - x1
    bh = y2 - y1

    # Enough context to see the object,
    # but much tighter than the first experiment
    padding = 0.40

    x1 -= bw * padding
    y1 -= bh * padding
    x2 += bw * padding
    y2 += bh * padding

    cap = cv2.VideoCapture(video_path)

    video_w = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    video_h = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(video_w, int(x2))
    y2 = min(video_h, int(y2))

    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w <= 0 or crop_h <= 0:
        print("❌ Invalid crop dimensions")
        cap.release()
        continue

    # Enlarge for visual inspection
    scale = 2

    panel_w = crop_w * scale
    panel_h = crop_h * scale

    # Make dimensions even for video codecs
    panel_w -= panel_w % 2
    panel_h -= panel_h % 2

    output_w = panel_w * 2
    output_h = panel_h

    output_path = os.path.join(
        OUTPUT_DIR,
        f"event_{event_num:02d}.mp4"
    )

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        12,
        (output_w, output_h)
    )

    if not writer.isOpened():
        print("❌ Could not create video writer")
        cap.release()
        continue

    # ChildPlay annotation frames are 1-based
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame - 1
    )

    written = 0

    for frame in range(
        start_frame,
        end_frame + 1
    ):

        ok, img = cap.read()

        if not ok:
            print(
                f"⚠️ Could not read frame {frame}"
            )
            break

        clean = img.copy()
        overlay = img.copy()

        bbox = np.asarray(
            bbox_data[0, frame - 1]
        ).astype(float)

        keypoints = np.asarray(
            kp_data[0, frame - 1]
        ).astype(float)

        scores = np.asarray(
            score_data[0, frame - 1]
        ).astype(float)

        # Bounding box
        if (
            len(bbox) == 4
            and np.all(np.isfinite(bbox))
        ):

            bx1, by1, bx2, by2 = bbox.astype(int)

            cv2.rectangle(
                overlay,
                (bx1, by1),
                (bx2, by2),
                (0,255,0),
                3
            )

        # Hand skeleton
        for a, b in connections:

            if (
                a < len(scores)
                and b < len(scores)
                and scores[a] > 0.3
                and scores[b] > 0.3
            ):

                pa = tuple(
                    keypoints[a].astype(int)
                )

                pb = tuple(
                    keypoints[b].astype(int)
                )

                cv2.line(
                    overlay,
                    pa,
                    pb,
                    (255,0,0),
                    2
                )

        # Landmarks
        for i, (x, y) in enumerate(keypoints):

            if (
                i < len(scores)
                and scores[i] > 0.3
            ):

                cv2.circle(
                    overlay,
                    (int(x), int(y)),
                    4,
                    (0,0,255),
                    -1
                )

                # Highlight thumb/index tips
                if i in [4, 8]:
                    cv2.putText(
                        overlay,
                        str(i),
                        (int(x)+5, int(y)-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0,255,255),
                        2
                    )

        clean_crop = clean[
            y1:y2,
            x1:x2
        ]

        overlay_crop = overlay[
            y1:y2,
            x1:x2
        ]

        clean_crop = cv2.resize(
            clean_crop,
            (panel_w, panel_h),
            interpolation=cv2.INTER_CUBIC
        )

        overlay_crop = cv2.resize(
            overlay_crop,
            (panel_w, panel_h),
            interpolation=cv2.INTER_CUBIC
        )

        cv2.putText(
            clean_crop,
            "CLEAN",
            (15,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

        cv2.putText(
            overlay_crop,
            "POSE OVERLAY",
            (15,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

        combined = np.hstack(
            [clean_crop, overlay_crop]
        )

        cv2.putText(
            combined,
            (
                f"Event {event_num:02d} | "
                f"{hand.upper()} | "
                f"Frame {frame}"
            ),
            (15, output_h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255,255,255),
            2
        )

        writer.write(combined)
        written += 1

    writer.release()
    cap.release()

    print(
        f"✅ Saved {output_path} "
        f"({written} frames)"
    )

    index_rows.append({
        "event_number": event_num,
        "video_file": f"event_{event_num:02d}.mp4",
        "video_id": row["video_id"],
        "clip": clip,
        "split": split,
        "pid": pid,
        "hand": hand,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "mean_visibility": row["mean_visibility"],
        "age_tier": row["age_tier"]
    })


index_df = pd.DataFrame(index_rows)

index_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "event_index.csv"
    ),
    index=False
)

print("\n======================================")
print("FINISHED")
print("======================================")
print("Generated:", len(index_df), "event videos")
print(
    "Index:",
    f"{OUTPUT_DIR}/event_index.csv"
)
