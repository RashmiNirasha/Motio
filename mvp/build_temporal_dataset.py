import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
from collections import Counter

# ----------------------------------------------------
# SETTINGS
# ----------------------------------------------------

ANNOT_ROOT = "data/childplay_hand/data/annotations"
POSE_ROOT = "data/childplay_hand/data/pose"
OUTPUT_DIR = "mvp/dataset"

SEQ_LEN = 20

# Minimum number of actual frames needed before
# we resample an event to 20 frames.
MIN_SEGMENT_LEN = 4

TARGET_LABELS = {
    "background": "background",
    "grasp": "grasp",
    "hold": "hold",
    "release": "release",
    "point": "point"
}

LABEL_NAMES = [
    "background",
    "grasp",
    "hold",
    "release",
    "point",
    "other"
]

LABEL_TO_ID = {
    name: i
    for i, name in enumerate(LABEL_NAMES)
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------
# Convert original ChildPlay interaction label
# ----------------------------------------------------

def map_label(label):

    label = str(label).strip().lower()

    if label in TARGET_LABELS:
        return TARGET_LABELS[label]

    # operate, put, give, drop, throw etc.
    return "other"


# ----------------------------------------------------
# Normalize one hand frame
# ----------------------------------------------------

def make_frame_features(
    keypoints,
    scores,
    bbox,
    hand
):

    keypoints = np.asarray(
        keypoints,
        dtype=np.float32
    )

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    bbox = np.asarray(
        bbox,
        dtype=np.float32
    )

    if keypoints.shape != (21, 2):
        return None

    if scores.shape[0] != 21:
        return None

    if bbox.shape[0] != 4:
        return None

    # Need a reasonably detected wrist
    wrist = keypoints[0].copy()

    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    scale = max(float(bw), float(bh), 1.0)

    # Wrist-relative coordinates
    relative = (
        keypoints - wrist
    ) / scale

    # Canonical orientation:
    # mirror left hand so left/right hands have
    # approximately the same coordinate convention.
    if str(hand).lower() == "left":
        relative[:, 0] *= -1.0

    # Suppress very unreliable joints
    reliable = scores >= 0.10

    relative[
        ~reliable
    ] = 0.0

    # x, y, confidence for each of 21 joints
    features = np.concatenate(
        [
            relative,
            scores[:, None]
        ],
        axis=1
    )

    # 21 × 3 = 63
    return features.reshape(-1)


# ----------------------------------------------------
# Resample arbitrary event to exactly SEQ_LEN frames
# ----------------------------------------------------

def resample_segment(frames, seq_len):

    frames = np.asarray(frames)

    indices = np.linspace(
        0,
        len(frames) - 1,
        seq_len
    )

    indices = np.rint(
        indices
    ).astype(int)

    return frames[indices]


# ----------------------------------------------------
# Split annotation stream into consecutive segments
# ----------------------------------------------------

def get_segments(df):

    df = df.sort_values("frame")

    segments = []

    current_frames = []
    current_label = None
    previous_frame = None

    for _, row in df.iterrows():

        frame = int(row["frame"])

        label = map_label(
            row["interaction"]
        )

        new_segment = (
            current_label is None
            or label != current_label
            or previous_frame is None
            or frame != previous_frame + 1
        )

        if new_segment:

            if current_frames:
                segments.append(
                    (
                        current_label,
                        current_frames
                    )
                )

            current_frames = [frame]
            current_label = label

        else:
            current_frames.append(frame)

        previous_frame = frame

    if current_frames:
        segments.append(
            (
                current_label,
                current_frames
            )
        )

    return segments


# ----------------------------------------------------
# Background / other can be extremely long.
# Break them into manageable chunks.
# ----------------------------------------------------

def make_event_windows(label, frames):

    frames = np.asarray(frames)

    if len(frames) < MIN_SEGMENT_LEN:
        return []

    # Short action event:
    # resample entire event.
    if len(frames) <= SEQ_LEN:

        return [
            resample_segment(
                frames,
                SEQ_LEN
            )
        ]

    windows = []

    # For long segments use overlapping windows.
    stride = 10

    start = 0

    while start + SEQ_LEN <= len(frames):

        windows.append(
            frames[
                start:start + SEQ_LEN
            ]
        )

        start += stride

    # Capture final portion if not already represented
    if (
        len(windows) == 0
        or windows[-1][-1] != frames[-1]
    ):
        windows.append(
            resample_segment(
                frames[-SEQ_LEN:],
                SEQ_LEN
            )
        )

    return windows


# ----------------------------------------------------
# Build one split
# ----------------------------------------------------

def build_split(split):

    print("\n" + "=" * 70)
    print("BUILDING:", split.upper())
    print("=" * 70)

    pose_path = os.path.join(
        POSE_ROOT,
        f"cp_pose_{split}.pkl"
    )

    print("Loading pose data:")
    print(pose_path)

    pose = joblib.load(
        pose_path
    )

    annotation_files = glob.glob(
        os.path.join(
            ANNOT_ROOT,
            split,
            "*.csv"
        )
    )

    print(
        "Annotation files:",
        len(annotation_files)
    )

    all_sequences = []
    all_labels = []
    all_metadata = []

    skipped_missing_pose = 0
    skipped_invalid = 0

    for file_num, csv_path in enumerate(
        annotation_files,
        start=1
    ):

        if (
            file_num == 1
            or file_num % 50 == 0
        ):
            print(
                f"Processing {file_num}/"
                f"{len(annotation_files)}"
            )

        df = pd.read_csv(
            csv_path
        )

        required = {
            "clip",
            "frame",
            "pid",
            "hand",
            "interaction"
        }

        if not required.issubset(
            df.columns
        ):
            continue

        # Usually one CSV corresponds to one clip,
        # but group robustly anyway.
        for (
            clip,
            pid,
            hand
        ), group in df.groupby(
            ["clip", "pid", "hand"]
        ):

            pid = int(pid)
            hand = str(hand).lower()

            if clip not in pose:
                skipped_missing_pose += 1
                continue

            if pid not in pose[clip]:
                skipped_missing_pose += 1
                continue

            pdata = pose[clip][pid]

            kp_key = (
                f"{hand}_hand_keypoint"
            )

            score_key = (
                f"{hand}_hand_keypoint_score"
            )

            bbox_key = (
                f"{hand}_hand_bbox"
            )

            if (
                kp_key not in pdata
                or score_key not in pdata
                or bbox_key not in pdata
            ):
                skipped_missing_pose += 1
                continue

            keypoints = np.asarray(
                pdata[kp_key]
            )

            scores = np.asarray(
                pdata[score_key]
            )

            boxes = np.asarray(
                pdata[bbox_key]
            )

            segments = get_segments(
                group
            )

            for (
                label,
                segment_frames
            ) in segments:

                event_windows = (
                    make_event_windows(
                        label,
                        segment_frames
                    )
                )

                for window in event_windows:

                    sequence = []

                    valid = True

                    for frame in window:

                        # ChildPlay frame numbers are 1-based
                        idx = int(frame) - 1

                        if (
                            idx < 0
                            or idx >= keypoints.shape[1]
                        ):
                            valid = False
                            break

                        features = make_frame_features(
                            keypoints[
                                0,
                                idx
                            ],
                            scores[
                                0,
                                idx
                            ],
                            boxes[
                                0,
                                idx
                            ],
                            hand
                        )

                        if features is None:
                            valid = False
                            break

                        sequence.append(
                            features
                        )

                    if not valid:
                        skipped_invalid += 1
                        continue

                    sequence = np.asarray(
                        sequence,
                        dtype=np.float32
                    )

                    # Expected:
                    # 20 frames × 63 features
                    if sequence.shape != (
                        SEQ_LEN,
                        63
                    ):
                        skipped_invalid += 1
                        continue

                    all_sequences.append(
                        sequence
                    )

                    all_labels.append(
                        LABEL_TO_ID[label]
                    )

                    all_metadata.append({
                        "split": split,
                        "clip": clip,
                        "pid": pid,
                        "hand": hand,
                        "start_frame": int(
                            window[0]
                        ),
                        "end_frame": int(
                            window[-1]
                        ),
                        "label": label
                    })

    X = np.asarray(
        all_sequences,
        dtype=np.float32
    )

    y = np.asarray(
        all_labels,
        dtype=np.int64
    )

    metadata = pd.DataFrame(
        all_metadata
    )

    print("\nResult shape:")
    print("X:", X.shape)
    print("y:", y.shape)

    counts = Counter(
        y.tolist()
    )

    print("\nClass counts:")

    for name, idx in LABEL_TO_ID.items():

        print(
            f"{name:12s}",
            counts.get(idx, 0)
        )

    print(
        "\nMissing-pose groups:",
        skipped_missing_pose
    )

    print(
        "Invalid sequences:",
        skipped_invalid
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"X_{split}.npy"
        ),
        X
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"y_{split}.npy"
        ),
        y
    )

    metadata.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"metadata_{split}.csv"
        ),
        index=False
    )

    return X, y


# ----------------------------------------------------
# BUILD ALL SPLITS
# ----------------------------------------------------

summary = {}

for split in [
    "train",
    "val",
    "test"
]:

    X, y = build_split(split)

    summary[split] = {
        "samples": int(len(y)),
        "shape": list(X.shape)
    }


with open(
    os.path.join(
        OUTPUT_DIR,
        "labels.json"
    ),
    "w"
) as f:

    json.dump(
        LABEL_TO_ID,
        f,
        indent=2
    )


with open(
    os.path.join(
        OUTPUT_DIR,
        "summary.json"
    ),
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    "Dataset saved to:",
    OUTPUT_DIR
)

print("\nLabel mapping:")

for name, idx in LABEL_TO_ID.items():

    print(
        idx,
        "=",
        name
    )
