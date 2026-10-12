import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
from collections import Counter

ANNOT_ROOT = "data/childplay_hand/data/annotations"
POSE_ROOT = "data/childplay_hand/data/pose"
OUTPUT_DIR = "mvp/dataset_v2"

SEQ_LEN = 20
MIN_SEGMENT_LEN = 4

LABEL_NAMES = [
    "background",
    "object_interaction",
    "point"
]

LABEL_TO_ID = {
    name: i
    for i, name in enumerate(LABEL_NAMES)
}

OBJECT_ACTIONS = {
    "grasp",
    "hold",
    "release",
    "operate",
    "put",
    "give",
    "drop",
    "throw"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def map_label(label):
    label = str(label).strip().lower()

    if label == "background":
        return "background"

    if label == "point":
        return "point"

    if label in OBJECT_ACTIONS:
        return "object_interaction"

    return "background"


def make_frame_features(
    keypoints,
    scores,
    bbox,
    hand,
    img_shape
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

    if len(scores) != 21:
        return None

    if len(bbox) != 4:
        return None

    img_h, img_w = img_shape[:2]

    wrist = keypoints[0].copy()

    x1, y1, x2, y2 = bbox

    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)

    scale = max(bw, bh, 1.0)

    # ---------------------------------------
    # LOCAL HAND SHAPE
    # ---------------------------------------

    relative = (
        keypoints - wrist
    ) / scale

    # Put left/right hands into similar orientation
    if str(hand).lower() == "left":
        relative[:, 0] *= -1.0

    reliable = scores >= 0.10

    relative[
        ~reliable
    ] = 0.0

    local_features = np.concatenate(
        [
            relative,
            scores[:, None]
        ],
        axis=1
    ).reshape(-1)

    # ---------------------------------------
    # GLOBAL POSITION + SCALE
    # ---------------------------------------

    wrist_x = wrist[0] / img_w
    wrist_y = wrist[1] / img_h

    center_x = (
        (x1 + x2) / 2
    ) / img_w

    center_y = (
        (y1 + y2) / 2
    ) / img_h

    norm_w = bw / img_w
    norm_h = bh / img_h

    global_features = np.asarray(
        [
            wrist_x,
            wrist_y,
            center_x,
            center_y,
            norm_w,
            norm_h
        ],
        dtype=np.float32
    )

    # 63 + 6 = 69 features
    return np.concatenate(
        [
            local_features,
            global_features
        ]
    )


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

            current_label = label
            current_frames = [frame]

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


def make_windows(frames):
    frames = np.asarray(frames)

    if len(frames) < MIN_SEGMENT_LEN:
        return []

    if len(frames) <= SEQ_LEN:
        return [
            resample_segment(
                frames,
                SEQ_LEN
            )
        ]

    windows = []

    # Less overlap than V1
    stride = 15

    for start in range(
        0,
        len(frames) - SEQ_LEN + 1,
        stride
    ):
        windows.append(
            frames[
                start:start + SEQ_LEN
            ]
        )

    return windows


def build_split(split):

    print("\n" + "=" * 70)
    print("BUILDING", split.upper())
    print("=" * 70)

    pose = joblib.load(
        os.path.join(
            POSE_ROOT,
            f"cp_pose_{split}.pkl"
        )
    )

    files = glob.glob(
        os.path.join(
            ANNOT_ROOT,
            split,
            "*.csv"
        )
    )

    X_all = []
    y_all = []
    metadata = []

    for file_num, path in enumerate(
        files,
        start=1
    ):

        if file_num == 1 or file_num % 50 == 0:
            print(
                f"Processing {file_num}/{len(files)}"
            )

        df = pd.read_csv(path)

        for (
            clip,
            pid,
            hand
        ), group in df.groupby(
            ["clip", "pid", "hand"]
        ):

            pid = int(pid)
            hand = str(hand).lower()

            if (
                clip not in pose
                or pid not in pose[clip]
            ):
                continue

            pdata = pose[clip][pid]

            kp = np.asarray(
                pdata[
                    f"{hand}_hand_keypoint"
                ]
            )

            scores = np.asarray(
                pdata[
                    f"{hand}_hand_keypoint_score"
                ]
            )

            boxes = np.asarray(
                pdata[
                    f"{hand}_hand_bbox"
                ]
            )

            img_shape = pdata["img_shape"]

            segments = get_segments(
                group
            )

            for label, frames in segments:

                windows = make_windows(
                    frames
                )

                for window in windows:

                    sequence = []
                    valid = True

                    for frame in window:

                        idx = int(frame) - 1

                        if (
                            idx < 0
                            or idx >= kp.shape[1]
                        ):
                            valid = False
                            break

                        features = make_frame_features(
                            kp[0, idx],
                            scores[0, idx],
                            boxes[0, idx],
                            hand,
                            img_shape
                        )

                        if features is None:
                            valid = False
                            break

                        sequence.append(
                            features
                        )

                    if not valid:
                        continue

                    sequence = np.asarray(
                        sequence,
                        dtype=np.float32
                    )

                    if sequence.shape != (
                        SEQ_LEN,
                        69
                    ):
                        continue

                    X_all.append(sequence)

                    y_all.append(
                        LABEL_TO_ID[label]
                    )

                    metadata.append({
                        "split": split,
                        "clip": clip,
                        "pid": pid,
                        "hand": hand,
                        "start_frame": int(window[0]),
                        "end_frame": int(window[-1]),
                        "label": label
                    })

    X = np.asarray(
        X_all,
        dtype=np.float32
    )

    y = np.asarray(
        y_all,
        dtype=np.int64
    )

    print("\nX:", X.shape)
    print("y:", y.shape)

    counts = Counter(y.tolist())

    print("\nCLASS COUNTS")

    for name, idx in LABEL_TO_ID.items():
        print(
            f"{name:20s}",
            counts.get(idx, 0)
        )

    np.save(
        f"{OUTPUT_DIR}/X_{split}.npy",
        X
    )

    np.save(
        f"{OUTPUT_DIR}/y_{split}.npy",
        y
    )

    pd.DataFrame(
        metadata
    ).to_csv(
        f"{OUTPUT_DIR}/metadata_{split}.csv",
        index=False
    )


for split in [
    "train",
    "val",
    "test"
]:
    build_split(split)


with open(
    f"{OUTPUT_DIR}/labels.json",
    "w"
) as f:

    json.dump(
        LABEL_TO_ID,
        f,
        indent=2
    )


print("\nDONE")
print("Saved to:", OUTPUT_DIR)
