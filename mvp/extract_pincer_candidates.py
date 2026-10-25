import argparse
import glob
import json
import os
import re
import shutil

import cv2
import joblib
import numpy as np
import pandas as pd

from age_hints import age_hint


# --------------------------------------------------
# Find ChildPlay-Hand moments that could show a
# pincer grasp, and save a zoomed hand crop of each
# for manual labelling in /app/label-pincer.html.
#
# ChildPlay has no grasp-type labels, so this only
# selects candidates: object-contact actions where
# the hand is large and its keypoints are confident.
# The pincer / not-pincer label comes from a person.
#
# Only children are kept (ChildPlay-Gaze is_child),
# and each video gets an age hint from its title for
# the labeller to confirm (ChildPlay has no ages).
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

ANNOT_ROOT = os.path.join(DATA_ROOT, "childplay_hand", "data", "annotations")
POSE_ROOT = os.path.join(DATA_ROOT, "childplay_hand", "data", "pose")
VIDEO_ROOT = os.path.join(DATA_ROOT, "ChildPlay-gaze", "videos")
CLIPS_CSV = os.path.join(DATA_ROOT, "ChildPlay-gaze", "clips.csv")
VIDEOS_CSV = os.path.join(DATA_ROOT, "ChildPlay-gaze", "videos.csv")
GAZE_ROOT = os.path.join(DATA_ROOT, "ChildPlay-gaze", "annotations")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "mvp", "pincer_candidates")

CONTACT_ACTIONS = {"grasp", "hold", "operate"}

THUMB_TIP = 4
INDEX_TIP = 8

# Same frame-offset quirk as ChildPlay-gaze's
# extract-clips-from-videos.py, so frames match
# the annotations.
SPECIAL_CLIPS = {
    "smwfiZd8HLc_7508-8408-downsampled", "smwfiZd8HLc_10346-12974-downsampled",
    "smwfiZd8HLc_16176-16586-downsampled", "smwfiZd8HLc_20322-20668-downsampled",
    "31lG75MDwSA_2857-2928", "31lG75MDwSA_3180-3306", "31lG75MDwSA_4686-4764",
    "EuYseDl2jm8_1869-2033", "EuYseDl2jm8_3680-3791"
}


parser = argparse.ArgumentParser()
parser.add_argument("--min-hand-px", type=float, default=90)
parser.add_argument("--min-score", type=float, default=0.6)
parser.add_argument("--min-frames", type=int, default=20)
parser.add_argument("--max-frames", type=int, default=50)
parser.add_argument("--crop-size", type=int, default=224)
parser.add_argument("--crop-scale", type=float, default=2.2)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument(
    "--include-adults",
    action="store_true",
    help="Also keep adult hands (default: children only)"
)
args = parser.parse_args()


for path in [ANNOT_ROOT, POSE_ROOT, CLIPS_CSV, GAZE_ROOT]:
    if not os.path.exists(path):
        raise SystemExit(
            f"Missing {path}\n"
            "Link or copy the ChildPlay data into data/ (see data/README.md)."
        )


# --------------------------------------------------
# Candidate selection from annotations + pose
# --------------------------------------------------

def good_frame(keypoints, scores):

    extent = np.ptp(keypoints, axis=0).max()

    return (
        extent >= args.min_hand_px
        and scores.mean() >= args.min_score
        and min(scores[THUMB_TIP], scores[INDEX_TIP]) >= args.min_score
    )


def runs_of(frames):

    # Split sorted frame numbers into consecutive runs.
    runs = []
    current = []

    for frame in frames:

        if current and frame != current[-1] + 1:
            runs.append(current)
            current = []

        current.append(frame)

    if current:
        runs.append(current)

    return runs


def chunk(run):

    # Long holds are cut into labelable chunks;
    # a short remainder is merged into the last chunk.
    chunks = [
        run[i:i + args.max_frames]
        for i in range(0, len(run), args.max_frames)
    ]

    if len(chunks) > 1 and len(chunks[-1]) < args.min_frames:
        remainder = chunks.pop()
        chunks[-1] = chunks[-1] + remainder

    return chunks


def load_is_child():

    # ChildPlay-Gaze person_id matches the hand
    # annotations' pid (same people, same boxes).
    is_child = {}

    for path in glob.glob(os.path.join(GAZE_ROOT, "*", "*.csv")):

        gaze = pd.read_csv(path, usecols=["clip", "person_id", "is_child"])

        for (clip, pid), values in gaze.groupby(["clip", "person_id"]):
            is_child[(clip, int(pid))] = bool(values.is_child.mean() > 0.5)

    return is_child


is_child = load_is_child()


def labelled_ranges():

    # Frame ranges you have already labelled, per hand track.
    # Changing --min-hand-px etc. moves chunk boundaries, so match
    # by overlap, not by candidate id.
    path = os.path.join(OUTPUT_DIR, "labels.json")

    if not os.path.exists(path):
        return {}

    with open(path) as f:
        labels = json.load(f)

    ranges = {}

    for candidate_id, entry in labels.items():

        if "start_frame" in entry:
            track = (entry["clip"], entry["pid"], entry["hand"])
            start, end = entry["start_frame"], entry["end_frame"]
        else:
            # Older labels only have the id: <clip>_p<pid>_<hand>_<start>.
            # Chunks are at most ~70 frames long.
            clip, pid, hand, start = re.match(
                r"(.+)_p(\d+)_(left|right)_(\d+)$", candidate_id
            ).groups()
            track = (clip, int(pid), hand)
            start = int(start)
            end = start + args.max_frames + args.min_frames - 2

        ranges.setdefault(track, []).append((start, end, candidate_id))

    return ranges


already_labelled = labelled_ranges()
skipped_labelled = 0

skipped_people = {"adult": set(), "unknown": set()}

candidates = []

for split in ["train", "val", "test"]:

    print(f"Scanning {split}...")

    pose = joblib.load(os.path.join(POSE_ROOT, f"cp_pose_{split}.pkl"))

    for path in sorted(glob.glob(os.path.join(ANNOT_ROOT, split, "*.csv"))):

        df = pd.read_csv(path)

        for (clip, pid, hand), group in df.groupby(["clip", "pid", "hand"]):

            pid = int(pid)
            hand = str(hand).lower()

            if clip not in pose or pid not in pose[clip]:
                continue

            child = is_child.get((clip, pid))

            if child is not True and not args.include_adults:
                skipped_people["adult" if child is False else "unknown"].add((clip, pid))
                continue

            pdata = pose[clip][pid]

            keypoints = np.asarray(pdata[f"{hand}_hand_keypoint"])[0]
            scores = np.asarray(pdata[f"{hand}_hand_keypoint_score"])[0]

            selected = sorted(
                int(row.frame)
                for row in group.itertuples()
                if str(row.interaction).lower() in CONTACT_ACTIONS
                and 0 < int(row.frame) <= len(keypoints)
                and good_frame(keypoints[int(row.frame) - 1], scores[int(row.frame) - 1])
            )

            for run in runs_of(selected):

                if len(run) < args.min_frames:
                    continue

                for part in chunk(run):

                    actions = (
                        group[group.frame.isin(part)]
                        .interaction.str.lower().value_counts().to_dict()
                    )

                    candidate_id = f"{clip}_p{pid}_{hand}_{part[0]}"

                    if any(
                        start <= part[-1] and end >= part[0]
                        and labelled_id != candidate_id
                        for start, end, labelled_id
                        in already_labelled.get((clip, pid, hand), [])
                    ):
                        skipped_labelled += 1
                        continue

                    candidates.append({
                        "candidate_id": candidate_id,
                        "split": split,
                        "clip": clip,
                        "video_id": clip[:11],
                        "pid": pid,
                        "hand": hand,
                        "is_child": child,
                        "start_frame": part[0],
                        "end_frame": part[-1],
                        "actions": actions,
                        "img_shape": [int(v) for v in pdata["img_shape"]],
                        "childplay_keypoints": keypoints[np.asarray(part) - 1].round(2).tolist()
                    })

print(
    f"\nSkipped {len(skipped_people['adult'])} adults and "
    f"{len(skipped_people['unknown'])} people without a child/adult annotation"
)

print(f"Skipped {skipped_labelled} chunks overlapping clips you already labelled")

print(f"Found {len(candidates)} candidate chunks "
      f"from {len({(c['clip'], c['pid'], c['hand']) for c in candidates})} hand tracks "
      f"in {len({c['video_id'] for c in candidates})} source videos")


# --------------------------------------------------
# Crop frames from the source videos
# --------------------------------------------------

clip_info = pd.read_csv(CLIPS_CSV).set_index("clip")

videos = pd.read_csv(VIDEOS_CSV).set_index("video_id")

for candidate in candidates:

    video_id = candidate["video_id"]

    if video_id in videos.index:
        row = videos.loc[video_id]
        candidate["video_title"] = row["title"]
        candidate["age_hint"], candidate["age_reasons"] = age_hint(
            row["title"], row["description"], row["keywords"]
        )
    else:
        candidate["video_title"] = ""
        candidate["age_hint"], candidate["age_reasons"] = "unknown", []

available = {
    os.path.splitext(name)[0]
    for name in os.listdir(VIDEO_ROOT)
} if os.path.isdir(VIDEO_ROOT) else set()


def video_frame_index(clip, clip_frame):

    # 0-based index in the source video for a 1-based clip frame,
    # following extract-clips-from-videos.py.
    interval = clip.rsplit("_", 1)[1]
    downsampled = "downsampled" in interval
    start = int(interval.replace("-downsampled", "").split("-")[0])

    index = start - 1 + (2 * clip_frame - 1 if downsampled else clip_frame - 1)

    if clip in SPECIAL_CLIPS:
        index -= 1

    return index


def smooth(values, window=5):

    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")

    return np.convolve(padded, kernel, mode="valid")


def extract(candidate, capture, out_dir, video_scale):

    keypoints = np.asarray(candidate["childplay_keypoints"]) * video_scale

    # Crop window follows the hand (smoothed),
    # with a fixed size for the whole chunk.
    centers = (keypoints.min(axis=1) + keypoints.max(axis=1)) / 2
    cx = smooth(centers[:, 0])
    cy = smooth(centers[:, 1])
    side = np.median(np.ptp(keypoints, axis=1).max(axis=1)) * args.crop_scale

    os.makedirs(out_dir, exist_ok=True)

    frames = range(candidate["start_frame"], candidate["end_frame"] + 1)

    for i, clip_frame in enumerate(frames):

        capture.set(cv2.CAP_PROP_POS_FRAMES, video_frame_index(candidate["clip"], clip_frame))
        ok, image = capture.read()

        if not ok:
            return False

        h, w = image.shape[:2]

        x0 = int(round(cx[i] - side / 2))
        y0 = int(round(cy[i] - side / 2))
        x1 = x0 + int(round(side))
        y1 = y0 + int(round(side))

        # Pad so crops near the image edge keep their size.
        pad = int(side)
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        crop = padded[y0 + pad:y1 + pad, x0 + pad:x1 + pad]

        cv2.imwrite(
            os.path.join(out_dir, f"frame_{i:03d}.jpg"),
            cv2.resize(crop, (args.crop_size, args.crop_size), interpolation=cv2.INTER_CUBIC),
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )

        if i == len(frames) // 2:

            # Whole-scene still, so the labeller can see
            # whose hand it is and what it holds.
            context = image.copy()
            cv2.rectangle(context, (x0, y0), (x1, y1), (0, 255, 0), max(2, w // 300))
            scale = 640 / w
            cv2.imwrite(
                os.path.join(out_dir, "context.jpg"),
                cv2.resize(context, (640, int(h * scale))),
                [cv2.IMWRITE_JPEG_QUALITY, 80]
            )

    candidate["n_frames"] = len(frames)

    return True


os.makedirs(OUTPUT_DIR, exist_ok=True)

extracted = []
missing_videos = set()

by_video = {}
for candidate in candidates:
    by_video.setdefault(candidate["video_id"], []).append(candidate)

for video_id, video_candidates in sorted(by_video.items()):

    if video_id not in available:
        missing_videos.add(video_id)
        continue

    video_path = glob.glob(os.path.join(VIDEO_ROOT, video_id + ".*"))[0]
    capture = cv2.VideoCapture(video_path)

    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = capture.get(cv2.CAP_PROP_FPS)

    print(f"{video_id}: {len(video_candidates)} candidates")

    for candidate in video_candidates:

        img_h, img_w = candidate["img_shape"]
        expected_fps = clip_info.loc[candidate["clip"], "fps"]

        if "downsampled" not in candidate["clip"] and abs(fps - expected_fps) > 0.5:
            print(f"  skip {candidate['candidate_id']}: video fps {fps:.2f} != annotation fps {expected_fps}")
            continue

        out_dir = os.path.join(OUTPUT_DIR, candidate["candidate_id"])

        if os.path.exists(os.path.join(out_dir, "context.jpg")) and not args.overwrite:
            candidate["n_frames"] = candidate["end_frame"] - candidate["start_frame"] + 1
            extracted.append(candidate)
            continue

        if extract(candidate, capture, out_dir, width / img_w):
            extracted.append(candidate)
        else:
            shutil.rmtree(out_dir, ignore_errors=True)
            print(f"  failed to read frames for {candidate['candidate_id']}")

    capture.release()


with open(os.path.join(OUTPUT_DIR, "candidates.json"), "w") as f:
    json.dump(extracted, f)


# Remove crops of candidates no longer selected
# (e.g. adults). Labels in labels.json are kept.
kept = {c["candidate_id"] for c in extracted}

for name in os.listdir(OUTPUT_DIR):

    path = os.path.join(OUTPUT_DIR, name)

    if os.path.isdir(path) and name not in kept:
        shutil.rmtree(path)


# Videos that still need downloading, with the
# resolution / fps the annotations were made at.
with open(os.path.join(OUTPUT_DIR, "missing_videos.csv"), "w") as f:

    f.write("video_id,height,fps,candidates\n")

    for video_id in sorted(missing_videos):

        height = videos.loc[video_id, "height"] if video_id in videos.index else ""
        fps = videos.loc[video_id, "fps"] if video_id in videos.index else ""
        f.write(f"{video_id},{height},{fps},{len(by_video[video_id])}\n")

print(f"\nExtracted {len(extracted)} candidates to {OUTPUT_DIR}")

if missing_videos:
    print(
        f"{len(missing_videos)} source videos are not downloaded "
        f"({sum(len(by_video[v]) for v in missing_videos)} more candidates). "
        "See missing_videos.csv and mvp/download_childplay_videos.sh."
    )
