import os
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np

POSE_PATH = "data/childplay_hand/data/pose/cp_pose_test.pkl"
IMAGE_ROOT = "data/ChildPlay-gaze/images"

pose = joblib.load(POSE_PATH)

events = [
    {
        "name": "event_A",
        "clip": "LHT2zVYvObg_1790-1917",
        "clip_start": 1790,
        "pid": 2,
        "hand": "right",
        "start": 51,
        "end": 70
    },
    {
        "name": "event_B",
        "clip": "LHT2zVYvObg_1921-2099",
        "clip_start": 1921,
        "pid": 1,
        "hand": "left",
        "start": 158,
        "end": 179
    }
]

def get_bbox(arr, frame):
    idx = frame - 1
    box = np.asarray(arr)

    if box.ndim == 3:
        box = box[0, idx]
    elif box.ndim == 2:
        box = box[idx]
    else:
        raise ValueError(f"Unexpected bbox shape: {box.shape}")

    return box.astype(float)

for event in events:
    clip = event["clip"]
    pid = event["pid"]
    side = event["hand"]
    bbox_key = f"{side}_hand_bbox"

    print("\n==============================")
    print(event["name"])
    print("==============================")
    print("Clip:", clip)
    print("PID:", pid)
    print("Hand:", side)

    pdata = pose[clip][pid]

    if bbox_key not in pdata:
        print("No hand bounding boxes available.")
        continue

    bbox_data = pdata[bbox_key]

    frames = np.linspace(event["start"], event["end"], 5).round().astype(int)

    fig, axes = plt.subplots(1, len(frames), figsize=(18, 4))

    for ax, local_frame in zip(axes, frames):
        original_frame = event["clip_start"] + local_frame - 1
        video_id = clip.split("_")[0]

        image_path = os.path.join(
            IMAGE_ROOT,
            clip,
            f"{video_id}_{original_frame}.jpg"
        )

        img = cv2.imread(image_path)

        if img is None:
            ax.set_title(f"Missing\n{local_frame}")
            ax.axis("off")
            continue

        h, w = img.shape[:2]
        x1, y1, x2, y2 = get_bbox(bbox_data, local_frame)

        bw = x2 - x1
        bh = y2 - y1

        padding = 1.0

        x1 = max(0, int(x1 - padding * bw))
        y1 = max(0, int(y1 - padding * bh))
        x2 = min(w, int(x2 + padding * bw))
        y2 = min(h, int(y2 + padding * bh))

        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            ax.set_title(f"Invalid crop\n{local_frame}")
            ax.axis("off")
            continue

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        ax.imshow(crop)
        ax.set_title(f"Frame {local_frame}\norig {original_frame}")
        ax.axis("off")

    plt.suptitle(f"{clip} | pid {pid} | {side.upper()} hand")
    plt.tight_layout()

    output = f"{event['name']}_hand_crops.png"
    plt.savefig(output, dpi=220, bbox_inches="tight")
    plt.close()

    print("Saved:", output)
