import os
import cv2
import joblib
import numpy as np

POSE_PATH = "data/childplay_hand/data/pose/cp_pose_test.pkl"
IMAGE_ROOT = "data/ChildPlay-gaze/images"

clip = "LHT2zVYvObg_1921-2099"
video_id = "LHT2zVYvObg"
clip_start = 1921

pid = 1
start_frame = 158
end_frame = 179

pose = joblib.load(POSE_PATH)

pdata = pose[clip][pid]

bbox_data = pdata["left_hand_bbox"]
keypoint_data = pdata["left_hand_keypoint"]
score_data = pdata["left_hand_keypoint_score"]

connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# First pass:
# determine one stable crop covering the whole event

all_boxes = []

for frame in range(start_frame, end_frame + 1):

    bbox = bbox_data[0, frame-1]

    if np.all(np.isfinite(bbox)):
        all_boxes.append(bbox)

all_boxes = np.asarray(all_boxes)

x1 = np.min(all_boxes[:,0])
y1 = np.min(all_boxes[:,1])
x2 = np.max(all_boxes[:,2])
y2 = np.max(all_boxes[:,3])

bw = x2 - x1
bh = y2 - y1

padding = 0.45

x1 = int(x1 - padding * bw)
y1 = int(y1 - padding * bh)
x2 = int(x2 + padding * bw)
y2 = int(y2 + padding * bh)

# Find first image for dimensions
orig = clip_start + start_frame - 1

sample_path = os.path.join(
    IMAGE_ROOT,
    clip,
    f"{video_id}_{orig}.jpg"
)

sample = cv2.imread(sample_path)

H, W = sample.shape[:2]

x1 = max(0, x1)
y1 = max(0, y1)
x2 = min(W, x2)
y2 = min(H, y2)

crop_w = x2 - x1
crop_h = y2 - y1

# Enlarge output to make fingers easier to see
scale = 3

output_size = (
    crop_w * scale,
    crop_h * scale
)

writer = cv2.VideoWriter(
    "event_B_grasp.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    12,
    output_size
)

for frame in range(start_frame, end_frame + 1):

    original_frame = (
        clip_start + frame - 1
    )

    path = os.path.join(
        IMAGE_ROOT,
        clip,
        f"{video_id}_{original_frame}.jpg"
    )

    img = cv2.imread(path)

    if img is None:
        continue

    bbox = bbox_data[0, frame-1]
    keypoints = keypoint_data[0, frame-1]
    scores = score_data[0, frame-1]

    # Draw hand bbox
    bx1, by1, bx2, by2 = bbox.astype(int)

    cv2.rectangle(
        img,
        (bx1, by1),
        (bx2, by2),
        (0,255,0),
        3
    )

    # Draw skeleton
    for a,b in connections:

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
                (255,0,0),
                2
            )

    for i,(x,y) in enumerate(keypoints):

        if scores[i] > 0.3:

            cv2.circle(
                img,
                (int(x),int(y)),
                5,
                (0,0,255),
                -1
            )

    crop = img[
        y1:y2,
        x1:x2
    ]

    crop = cv2.resize(
        crop,
        output_size,
        interpolation=cv2.INTER_CUBIC
    )

    cv2.putText(
        crop,
        f"Frame {frame}",
        (15,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    writer.write(crop)

writer.release()

print("Saved: event_B_grasp.mp4")
