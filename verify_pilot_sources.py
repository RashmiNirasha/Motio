import os
import cv2
import pandas as pd

sources = pd.read_csv("pilot_source_videos.csv")

print("\nPILOT SOURCE VIDEO VERIFICATION")
print("=" * 85)

all_ok = True

for _, row in sources.iterrows():

    vid = row["video_id"]
    path = f"data/ChildPlay-gaze/videos/{vid}.mp4"

    print("\n", vid)

    if not os.path.exists(path):
        print("  ❌ FILE MISSING")
        all_ok = False
        continue

    cap = cv2.VideoCapture(path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    expected_fps = float(row["fps"])
    expected_frames = int(row["frame_count"])
    expected_width = int(row["width"])
    expected_height = int(row["height"])

    fps_ok = abs(fps - expected_fps) < 0.1
    frame_diff = abs(frames - expected_frames)
    frames_ok = frame_diff <= 1
    res_ok = (
        width == expected_width and
        height == expected_height
    )

    print(
        f"  FPS:        {fps:.2f} "
        f"(expected {expected_fps}) "
        f"{'✅' if fps_ok else '❌'}"
    )

    print(
        f"  Frames:     {frames} "
        f"(expected {expected_frames}) "
        f"{'✅' if frames_ok else '❌'}"
    )

    print(
        f"  Resolution: {width}x{height} "
        f"(expected {expected_width}x{expected_height}) "
        f"{'✅' if res_ok else '❌'}"
    )

    if not (fps_ok and frames_ok and res_ok):
        all_ok = False

print("\n==========================================")

if all_ok:
    print("✅ ALL PILOT SOURCE VIDEOS MATCH")
else:
    print("⚠️ SOME VIDEOS DO NOT MATCH")
    print("Do NOT extract clips from mismatched sources yet.")
