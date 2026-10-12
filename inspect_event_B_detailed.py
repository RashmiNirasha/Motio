import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt

POSE_PATH = "data/childplay_hand/data/pose/cp_pose_test.pkl"
IMAGE_ROOT = "data/ChildPlay-gaze/images"

clip = "LHT2zVYvObg_1921-2099"
video_id = "LHT2zVYvObg"
clip_start = 1921

pid = 1
hand = "left"

start_frame = 158
end_frame = 179

pose = joblib.load(POSE_PATH)
pdata = pose[clip][pid]

bbox_data = pdata["left_hand_bbox"]
keypoint_data = pdata["left_hand_keypoint"]
score_data = pdata["left_hand_keypoint_score"]


def get_bbox(frame):
    idx = frame - 1
    return np.asarray(
        bbox_data[0, idx]
    ).astype(float)


def get_keypoints(frame):
    idx = frame - 1
    return np.asarray(
        keypoint_data[0, idx]
    ).astype(float)


def get_scores(frame):
    idx = frame - 1
    return np.asarray(
        score_data[0, idx]
    ).astype(float)


# Hand landmark connections
connections = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (5,9),(9,10),(10,11),(11,12),  # middle
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]


selected_frames = [158, 163, 168, 174, 179]

fig, axes = plt.subplots(
    1,
    len(selected_frames),
    figsize=(18, 4)
)

for ax, local_frame in zip(axes, selected_frames):

    original_frame = (
        clip_start + local_frame - 1
    )

    path = os.path.join(
        IMAGE_ROOT,
        clip,
        f"{video_id}_{original_frame}.jpg"
    )

    img = cv2.imread(path)

    if img is None:
        ax.axis("off")
        continue

    x1, y1, x2, y2 = get_bbox(local_frame)

    # Much tighter padding than before
    bw = x2 - x1
    bh = y2 - y1
    padding = 0.35

    cx1 = max(0, int(x1 - padding * bw))
    cy1 = max(0, int(y1 - padding * bh))
    cx2 = min(img.shape[1], int(x2 + padding * bw))
    cy2 = min(img.shape[0], int(y2 + padding * bh))

    # Draw bbox on original image
    cv2.rectangle(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 255, 0),
        3
    )

    keypoints = get_keypoints(local_frame)
    scores = get_scores(local_frame)

    # Draw connections
    for a, b in connections:
        if scores[a] > 0.3 and scores[b] > 0.3:
            pa = tuple(
                keypoints[a].astype(int)
            )
            pb = tuple(
                keypoints[b].astype(int)
            )

            cv2.line(
                img,
                pa,
                pb,
                (255, 0, 0),
                2
            )

    # Draw landmarks
    for i, (x, y) in enumerate(keypoints):

        if scores[i] > 0.3:

            cv2.circle(
                img,
                (int(x), int(y)),
                4,
                (0, 0, 255),
                -1
            )

            # Label thumb tip and index tip
            if i in [4, 8]:

                cv2.putText(
                    img,
                    str(i),
                    (int(x)+5, int(y)-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )

    crop = img[cy1:cy2, cx1:cx2]

    crop = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    ax.imshow(crop)

    ax.set_title(
        f"Frame {local_frame}\n"
        f"Original {original_frame}"
    )

    ax.axis("off")

plt.suptitle(
    "Event B — Child pid 1 — LEFT hand\n"
    "Red dots = landmarks | 4 = thumb tip | 8 = index tip"
)

plt.tight_layout()

plt.savefig(
    "event_B_detailed.png",
    dpi=250,
    bbox_inches="tight"
)

print("Saved: event_B_detailed.png")
