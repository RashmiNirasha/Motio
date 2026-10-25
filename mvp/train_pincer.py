import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(
    0,
    os.path.dirname(__file__)
)

from model_pincer import PincerTemporalNet
from pincer_features import (
    FEATURE_DIM,
    LABELS,
    SEQUENCE_LENGTH,
    heuristic_pincer_probability,
    landmarks_to_features
)


# --------------------------------------------------
# Train a pincer-grasp classifier on clips recorded
# with /app/record-pincer.html
#
#   1 = pincer
#   0 = other_grasp or no_grasp
#
# Clips from the same recording session / source
# video stay in the same split, so overlapping
# windows don't leak between train and evaluation.
#
# Modes:
#   default            train / val / test split by video
#   --cross-validate   leave-one-video-out evaluation, then
#                      a final model trained on all videos
#                      (better when there are few videos)
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

DEFAULT_DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "mvp",
    "pincer_data",
    "raw"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "mvp",
    "models_pincer"
)


parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
parser.add_argument("--model-dir", default=MODEL_DIR)
parser.add_argument(
    "--epochs",
    type=int,
    default=None,
    help="Default 80 with early stopping, 30 with --cross-validate"
)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--val-fraction", type=float, default=0.2)
parser.add_argument("--test-fraction", type=float, default=0.2)
parser.add_argument(
    "--under-5-only",
    action="store_true",
    help="Keep only ChildPlay videos confirmed as under 5 in the labeller"
)
parser.add_argument(
    "--cross-validate",
    action="store_true",
    help="Leave-one-video-out evaluation, then train on all videos"
)
parser.add_argument(
    "--no-save",
    action="store_true",
    help="Only evaluate; keep the current saved model"
)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if args.epochs is None:
    args.epochs = 30 if args.cross_validate else 80

rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)


# --------------------------------------------------
# Load recorded clips
# --------------------------------------------------

CANDIDATE_DIR = os.path.join(PROJECT_ROOT, "mvp", "pincer_candidates")


def load_json(path, default):

    if not os.path.exists(path):
        return default

    with open(path) as f:
        return json.load(f)


# ChildPlay clips are only used while their candidate
# is still selected (children only), and, with
# --under-5-only, when the video is confirmed under 5.
child_candidates = {
    c["candidate_id"]
    for c in load_json(os.path.join(CANDIDATE_DIR, "candidates.json"), [])
}

video_ages = load_json(os.path.join(CANDIDATE_DIR, "video_age.json"), {})

# Labelled clips stay valid after re-extraction changes candidate ids.
labelled_ids = set(load_json(os.path.join(CANDIDATE_DIR, "labels.json"), {}))

skipped = {"not a child candidate": 0, "not confirmed under 5": 0}

clips = []

for path in sorted(glob.glob(os.path.join(args.data_dir, "*.json"))):

    with open(path, "r") as f:
        clip = json.load(f)

    frames = np.asarray(clip["landmarks"], dtype=np.float32)

    if len(frames) < SEQUENCE_LENGTH:
        continue

    if clip.get("source") == "childplay":

        if not (
            clip.get("is_child")
            or clip["source_name"] in child_candidates
            or clip["source_name"] in labelled_ids
        ):
            skipped["not a child candidate"] += 1
            continue

        video_id = clip["session_id"][len("childplay_"):]

        if args.under_5_only and video_ages.get(video_id) != "under5":
            skipped["not confirmed under 5"] += 1
            continue

    clips.append({
        "label": clip["label"],
        "group": clip["session_id"],
        "frames": frames
    })

for reason, count in skipped.items():
    if count:
        print(f"  skipped {count} ChildPlay clips: {reason}")

if not clips and skipped["not confirmed under 5"]:
    sys.exit(
        "No clips left after --under-5-only. Confirm video ages "
        "(Y / O / N) in /app/label-pincer.html first."
    )

if not clips:
    sys.exit(
        f"No usable clips found in {args.data_dir}.\n"
        "Label ChildPlay candidates in /app/label-pincer.html or "
        "record clips with /app/record-pincer.html first."
    )

print(f"Loaded {len(clips)} clips")


for label in LABELS:
    selected = [c for c in clips if c["label"] == label]
    print(
        f"  {label:12s} clips={len(selected):4d} "
        f"frames={sum(len(c['frames']) for c in selected):6d}"
    )

groups = sorted({c["group"] for c in clips})

print("\nClips per session / video:")

for group in groups:
    labels = [c["label"] for c in clips if c["group"] == group]
    print(
        f"  {group:28s} "
        + "  ".join(f"{label} {labels.count(label)}" for label in LABELS)
    )


# --------------------------------------------------
# Windows, features, augmentation
# --------------------------------------------------

def make_windows(selected_clips, stride):

    windows = []
    labels = []
    fine_labels = []
    clip_index = []

    for index, clip in enumerate(selected_clips):

        frames = clip["frames"]

        for start in range(0, len(frames) - SEQUENCE_LENGTH + 1, stride):

            windows.append(frames[start:start + SEQUENCE_LENGTH])
            labels.append(1.0 if clip["label"] == "pincer" else 0.0)
            fine_labels.append(clip["label"])
            clip_index.append(index)

    return (
        np.asarray(windows, dtype=np.float32).reshape(-1, SEQUENCE_LENGTH, 21, 3),
        np.asarray(labels, dtype=np.float32),
        np.asarray(fine_labels),
        np.asarray(clip_index)
    )


def windows_to_features(windows):

    n = len(windows)

    return landmarks_to_features(
        windows.reshape(n * SEQUENCE_LENGTH, 21, 3)
    ).reshape(n, SEQUENCE_LENGTH, -1)


def augment(windows):

    # Pincer is the same for left and right hands and
    # for small camera rotations, so randomly mirror
    # and rotate each window in the image plane.
    windows = windows.copy()

    n = len(windows)

    mirror = rng.random(n) < 0.5
    windows[mirror, :, :, 0] *= -1

    angle = rng.uniform(-np.pi / 9, np.pi / 9, size=n)
    cos = np.cos(angle)[:, None, None]
    sin = np.sin(angle)[:, None, None]

    x = windows[..., 0].copy()
    y = windows[..., 1].copy()

    windows[..., 0] = cos * x - sin * y
    windows[..., 1] = sin * x + cos * y

    return windows


# --------------------------------------------------
# Training
# --------------------------------------------------

def train_model(L_train, y_train, L_val=None, y_val=None, verbose=True):

    # With validation data: keep the epoch with the
    # lowest validation loss. Without: fixed epochs.

    mirrored = L_train.copy()
    mirrored[..., 0] *= -1

    normalization_features = windows_to_features(
        np.concatenate([L_train, mirrored])
    ).reshape(-1, FEATURE_DIM)

    feature_mean = normalization_features.mean(axis=0)
    feature_std = normalization_features.std(axis=0) + 1e-6

    model = PincerTemporalNet(feature_mean, feature_std)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-3
    )

    pos_weight = torch.tensor(
        (len(y_train) - y_train.sum()) / y_train.sum(),
        dtype=torch.float32
    )

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    use_val = L_val is not None and len(L_val) > 0

    if use_val:
        X_val = torch.from_numpy(windows_to_features(L_val))
        Y_val = torch.from_numpy(y_val)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):

        model.train()

        X_train = torch.from_numpy(windows_to_features(augment(L_train)))
        Y_train = torch.from_numpy(y_train)

        order = torch.randperm(len(Y_train))
        train_loss = 0.0

        for start in range(0, len(order), args.batch_size):

            batch = order[start:start + args.batch_size]

            optimizer.zero_grad()
            loss = loss_fn(model(X_train[batch]), Y_train[batch])
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch)

        train_loss /= len(order)

        val_loss = float("nan")

        if use_val:

            model.eval()

            with torch.no_grad():
                val_loss = loss_fn(model(X_val), Y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(
                f"Epoch {epoch:3d}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    return model


def predict(model, windows):

    if not len(windows):
        return np.zeros(0)

    with torch.no_grad():
        return torch.sigmoid(
            model(torch.from_numpy(windows_to_features(windows)))
        ).numpy()


def heuristic_scores(windows):

    return np.asarray([
        heuristic_pincer_probability(window)
        for window in windows
    ])


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def metrics(scores, labels, threshold):

    predicted = scores >= threshold
    actual = labels == 1

    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    tn = int(np.sum(~predicted & ~actual))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "accuracy": (tp + tn) / len(labels) if len(labels) else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": [[tn, fp], [fn, tp]]
    }


def report(name, scores, labels, fine_labels, threshold):

    result = metrics(scores, labels, threshold)

    print(f"\n{name} at threshold {threshold:.2f}")
    print(f"  Accuracy  {result['accuracy']:.3f}")
    print(f"  Precision {result['precision']:.3f}")
    print(f"  Recall    {result['recall']:.3f}")
    print(f"  F1        {result['f1']:.3f}")
    print(f"  Confusion [[TN, FP], [FN, TP]] = {result['confusion']}")

    print("  Share predicted as pincer, by label:")

    for label in LABELS:

        mask = fine_labels == label

        if mask.any():
            share = np.mean(scores[mask] >= threshold)
            print(f"    {label:12s} {share:.3f}  ({int(mask.sum())} windows)")

    return result


def tune_threshold(scores, labels):

    threshold = 0.5
    best_f1 = -1.0

    for candidate in np.arange(0.05, 0.96, 0.01):

        f1 = metrics(scores, labels, candidate)["f1"]

        # On ties, prefer the threshold nearest 0.5.
        if f1 > best_f1 + 1e-9 or (
            abs(f1 - best_f1) <= 1e-9
            and abs(candidate - 0.5) < abs(threshold - 0.5)
        ):
            best_f1 = f1
            threshold = float(round(candidate, 2))

    return threshold


def has_both_classes(selected_groups):

    labels = {
        c["label"] == "pincer"
        for c in clips
        if c["group"] in selected_groups
    }

    return labels == {True, False}


def clips_in(selected_groups):

    return [c for c in clips if c["group"] in selected_groups]


# --------------------------------------------------
# Leave-one-video-out cross-validation
#
# Every video is the test set once, with a model
# trained on all the other videos. Pooling the
# held-out predictions uses all the data for an
# honest estimate, even with only a few videos.
# --------------------------------------------------

def cross_validate():

    pooled = {"model": [], "rule": [], "labels": [], "fine": []}
    clip_rows = []

    print(f"\nLeave-one-video-out over {len(groups)} sessions/videos "
          f"({args.epochs} epochs per fold, threshold 0.50)")

    for group in groups:

        train_groups = set(groups) - {group}

        L_train, y_train, _, _ = make_windows(clips_in(train_groups), stride=2)
        test_clips = clips_in({group})
        L_test, y_test, fine_test, clip_index = make_windows(test_clips, stride=5)

        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            print(f"  {group}: skipped (training videos lack one of the classes)")
            continue

        model = train_model(L_train, y_train, verbose=False)

        model_scores = predict(model, L_test)
        rule_scores = heuristic_scores(L_test)

        pooled["model"].append(model_scores)
        pooled["rule"].append(rule_scores)
        pooled["labels"].append(y_test)
        pooled["fine"].append(fine_test)

        for index, clip in enumerate(test_clips):

            mask = clip_index == index

            clip_rows.append({
                "pincer": clip["label"] == "pincer",
                "model": float(model_scores[mask].mean()),
                "rule": float(rule_scores[mask].mean())
            })

        fold = metrics(model_scores, y_test, 0.5)
        print(
            f"  test {group:28s} windows={len(y_test):4d} "
            f"pincer={int(y_test.sum()):3d}  "
            f"model acc={fold['accuracy']:.2f} recall={fold['recall']:.2f}"
        )

    if not pooled["labels"]:
        sys.exit("No usable folds. Label pincer clips in more videos.")

    labels = np.concatenate(pooled["labels"])
    fine = np.concatenate(pooled["fine"])

    result = report(
        "CROSS-VALIDATED model, all held-out windows (the honest number)",
        np.concatenate(pooled["model"]),
        labels,
        fine,
        0.5
    )

    report(
        "Rule-based heuristic on the same windows (baseline to beat)",
        np.concatenate(pooled["rule"]),
        labels,
        fine,
        0.5
    )

    clip_labels = np.asarray([r["pincer"] for r in clip_rows], dtype=np.float32)

    for name in ["model", "rule"]:

        clip_result = metrics(
            np.asarray([r[name] for r in clip_rows]),
            clip_labels,
            0.5
        )

        print(
            f"\nPer clip ({name}, mean score over its windows): "
            f"accuracy {clip_result['accuracy']:.3f}  "
            f"precision {clip_result['precision']:.3f}  "
            f"recall {clip_result['recall']:.3f}  "
            f"confusion {clip_result['confusion']}"
        )

    n_pincer = int(clip_labels.sum())

    if n_pincer < 30:
        print(
            f"\nNote: only {n_pincer} pincer clips were evaluated, so these "
            "numbers can swing a lot with a few more labels."
        )

    return result


# --------------------------------------------------
# Run
# --------------------------------------------------

if args.cross_validate:

    if len(groups) < 2:
        sys.exit("Cross-validation needs at least 2 sessions / source videos.")

    cv_result = cross_validate()

    # Final model: all videos, same fixed recipe that
    # was just evaluated.
    L_all, y_all, _, _ = make_windows(clips, stride=2)

    if y_all.sum() == 0 or y_all.sum() == len(y_all):
        sys.exit("Need both pincer and non-pincer clips to train.")

    print(f"\nTraining final model on all {len(groups)} sessions/videos...")

    model = train_model(L_all, y_all)
    threshold = 0.5

    split_info = {
        "evaluation": "leave-one-video-out",
        "train_sessions": groups,
        "test_metrics": cv_result
    }

else:

    # ----------------------------------------------
    # Train / val / test split by session / video
    #
    #   train: fit the model
    #   val:   early stopping + threshold
    #   test:  untouched until the end
    # ----------------------------------------------

    if len(groups) < 3:
        sys.exit(
            "Need at least 3 sessions / source videos for train, validation "
            "and test splits. Try --cross-validate."
        )

    n_test = max(1, int(round(len(groups) * args.test_fraction)))
    n_val = max(1, int(round(len(groups) * args.val_fraction)))

    for attempt in range(500):

        order = list(rng.permutation(groups))
        test_groups = set(order[:n_test])
        val_groups = set(order[n_test:n_test + n_val])
        train_groups = set(order[n_test + n_val:])

        if all(has_both_classes(g) for g in [train_groups, val_groups, test_groups]):
            break

    else:
        print(
            "WARNING: could not find a split where train, validation and "
            "test all contain both pincer and non-pincer clips. Label more "
            "videos, or use --cross-validate."
        )

    L_train, y_train, _, _ = make_windows(clips_in(train_groups), stride=2)
    L_val, y_val, fine_val, _ = make_windows(clips_in(val_groups), stride=5)
    L_test, y_test, fine_test, _ = make_windows(clips_in(test_groups), stride=5)

    print(f"\nTrain windows: {len(y_train)} (pincer {int(y_train.sum())})")
    print(f"Val windows:   {len(y_val)} (pincer {int(y_val.sum())})")
    print(
        f"Test windows:  {len(y_test)} (pincer {int(y_test.sum())}) "
        f"from {len(test_groups)} held-out sessions/videos"
    )

    if y_train.sum() == 0 or y_train.sum() == len(y_train):
        sys.exit("Training split needs both pincer and non-pincer clips.")

    model = train_model(L_train, y_train, L_val, y_val)

    threshold = 0.5

    if len(y_val):
        val_scores = predict(model, L_val)
        threshold = tune_threshold(val_scores, y_val)
        report("Validation (threshold tuned here, optimistic)", val_scores, y_val, fine_val, threshold)

    test_result = None

    if len(y_test):
        test_result = report(
            "TEST (held-out sessions/videos, the honest number)",
            predict(model, L_test),
            y_test,
            fine_test,
            threshold
        )

    split_info = {
        "evaluation": "train/val/test split",
        "train_sessions": sorted(train_groups),
        "val_sessions": sorted(val_groups),
        "test_sessions": sorted(test_groups),
        "test_metrics": test_result
    }


# --------------------------------------------------
# Save
# --------------------------------------------------

if args.no_save:
    print("\n--no-save: model not saved.")
    sys.exit(0)

os.makedirs(args.model_dir, exist_ok=True)

model_path = os.path.join(args.model_dir, "pincer_model.pt")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "feature_mean": model.feature_mean.reshape(-1).numpy(),
        "feature_std": model.feature_std.reshape(-1).numpy(),
        "sequence_length": SEQUENCE_LENGTH,
        "threshold": threshold,
        **split_info
    },
    model_path
)

with open(os.path.join(args.model_dir, "threshold.txt"), "w") as f:
    f.write(f"{threshold:.2f}\n")

print(f"\nSaved model to {model_path}")
print("Restart the FastAPI server to use it.")
