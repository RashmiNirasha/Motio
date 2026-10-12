import sys
import os
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)

sys.path.append("mvp")

from model_binary import BinaryTemporalHandNet


DATA_DIR = "mvp/dataset_v2"
MODEL_PATH = "mvp/models_binary/child_object_interaction.pt"

TARGET_PRECISION = 0.70


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)


# --------------------------------------------------
# LOAD CHECKPOINT
# --------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

feature_mean = checkpoint["feature_mean"]
feature_std = checkpoint["feature_std"]


model = BinaryTemporalHandNet(
    feature_mean,
    feature_std
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


# --------------------------------------------------
# LOAD SPLIT
# --------------------------------------------------

def load_split(split):

    X = np.load(
        f"{DATA_DIR}/X_{split}.npy"
    ).astype(np.float32)

    original_y = np.load(
        f"{DATA_DIR}/y_{split}.npy"
    )

    # object_interaction = original class 1
    y = (
        original_y == 1
    ).astype(np.int64)

    return X, y


@torch.no_grad()
def get_scores(split):

    X, y = load_split(split)

    dataset = TensorDataset(
        torch.from_numpy(X)
    )

    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False
    )

    scores = []

    for (batch,) in loader:

        batch = batch.to(device)

        logits = model(batch)

        probabilities = torch.sigmoid(
            logits
        )

        scores.extend(
            probabilities
            .cpu()
            .numpy()
            .tolist()
        )

    return (
        y,
        np.asarray(scores)
    )


# --------------------------------------------------
# VALIDATION THRESHOLD SEARCH
# --------------------------------------------------

val_y, val_scores = get_scores("val")

results = []

print("\nVALIDATION THRESHOLD SWEEP")
print("=" * 70)

print(
    f"{'Threshold':>10} "
    f"{'Precision':>10} "
    f"{'Recall':>10} "
    f"{'F1':>10}"
)

for threshold in np.arange(
    0.30,
    0.91,
    0.02
):

    pred = (
        val_scores >= threshold
    ).astype(int)

    precision = precision_score(
        val_y,
        pred,
        zero_division=0
    )

    recall = recall_score(
        val_y,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        val_y,
        pred,
        zero_division=0
    )

    results.append(
        (
            threshold,
            precision,
            recall,
            f1
        )
    )

    print(
        f"{threshold:10.2f} "
        f"{precision:10.3f} "
        f"{recall:10.3f} "
        f"{f1:10.3f}"
    )


# --------------------------------------------------
# Choose highest recall with precision >= target
# --------------------------------------------------

eligible = [
    r
    for r in results
    if r[1] >= TARGET_PRECISION
]

if eligible:

    selected = max(
        eligible,
        key=lambda r: r[2]
    )

    reason = (
        f"highest recall with "
        f"precision >= {TARGET_PRECISION:.2f}"
    )

else:

    selected = max(
        results,
        key=lambda r: r[3]
    )

    reason = "best validation F1"


threshold = float(
    selected[0]
)


print("\n" + "=" * 70)
print("SELECTED OPERATING POINT")
print("=" * 70)

print("Reason:", reason)

print(
    f"Threshold: {selected[0]:.2f}"
)

print(
    f"Val precision: {selected[1]:.3f}"
)

print(
    f"Val recall:    {selected[2]:.3f}"
)

print(
    f"Val F1:        {selected[3]:.3f}"
)


# --------------------------------------------------
# ONE FINAL TEST EVALUATION
# --------------------------------------------------

test_y, test_scores = get_scores("test")

test_pred = (
    test_scores >= threshold
).astype(int)


precision = precision_score(
    test_y,
    test_pred,
    zero_division=0
)

recall = recall_score(
    test_y,
    test_pred,
    zero_division=0
)

f1 = f1_score(
    test_y,
    test_pred,
    zero_division=0
)

accuracy = accuracy_score(
    test_y,
    test_pred
)

cm = confusion_matrix(
    test_y,
    test_pred
)


print("\n" + "=" * 70)
print("TEST AT SELECTED THRESHOLD")
print("=" * 70)

print(
    f"Threshold: {threshold:.2f}"
)

print(
    f"Accuracy:  {accuracy:.3f}"
)

print(
    f"Precision: {precision:.3f}"
)

print(
    f"Recall:    {recall:.3f}"
)

print(
    f"F1:        {f1:.3f}"
)

print("\nConfusion matrix:")

print(cm)


# Save threshold for iPhone
with open(
    "mvp/models_binary/iphone_threshold.txt",
    "w"
) as f:

    f.write(
        f"{threshold:.4f}"
    )

print(
    "\nSaved threshold to "
    "mvp/models_binary/iphone_threshold.txt"
)
