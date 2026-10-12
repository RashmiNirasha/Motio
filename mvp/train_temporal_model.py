import os
import sys
import copy
import json

import numpy as np
import pandas as pd
import torch

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    WeightedRandomSampler
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)

import matplotlib.pyplot as plt

sys.path.append("mvp")

from model import TemporalHandNet


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DATA_DIR = "mvp/dataset"
MODEL_DIR = "mvp/models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

LABELS = [
    "background",
    "grasp",
    "hold",
    "release",
    "point",
    "other"
]

NUM_CLASSES = len(LABELS)

BATCH_SIZE = 256
MAX_EPOCHS = 30
PATIENCE = 6
LEARNING_RATE = 1e-3

np.random.seed(42)
torch.manual_seed(42)


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print("\nUsing device:", device)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_split(split):

    X = np.load(
        os.path.join(
            DATA_DIR,
            f"X_{split}.npy"
        )
    ).astype(np.float32)

    y = np.load(
        os.path.join(
            DATA_DIR,
            f"y_{split}.npy"
        )
    ).astype(np.int64)

    return X, y


X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test")


print("\nDataset sizes:")

print(
    "Train:",
    X_train.shape
)

print(
    "Val:  ",
    X_val.shape
)

print(
    "Test: ",
    X_test.shape
)


# --------------------------------------------------
# TRAINING-SET NORMALIZATION
# --------------------------------------------------

feature_mean = X_train.mean(
    axis=(0, 1)
)

feature_std = X_train.std(
    axis=(0, 1)
)

# Avoid divide-by-zero
feature_std[
    feature_std < 1e-6
] = 1.0


np.savez(
    os.path.join(
        MODEL_DIR,
        "normalization.npz"
    ),
    mean=feature_mean,
    std=feature_std
)


# --------------------------------------------------
# DATASETS
# --------------------------------------------------

train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train)
)

val_dataset = TensorDataset(
    torch.from_numpy(X_val),
    torch.from_numpy(y_val)
)

test_dataset = TensorDataset(
    torch.from_numpy(X_test),
    torch.from_numpy(y_test)
)


# --------------------------------------------------
# BALANCED TRAINING SAMPLER
# --------------------------------------------------

class_counts = np.bincount(
    y_train,
    minlength=NUM_CLASSES
)

print("\nTraining class counts:")

for i, name in enumerate(LABELS):

    print(
        f"{name:12s}",
        class_counts[i]
    )


class_weights = (
    1.0 /
    np.maximum(
        class_counts,
        1
    )
)

sample_weights = (
    class_weights[
        y_train
    ]
)

sampler = WeightedRandomSampler(
    weights=torch.as_tensor(
        sample_weights,
        dtype=torch.double
    ),
    num_samples=len(y_train),
    replacement=True
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=512,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=512,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = TemporalHandNet(
    feature_mean=feature_mean,
    feature_std=feature_std,
    num_classes=NUM_CLASSES
).to(device)


num_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "\nTrainable parameters:",
    f"{num_params:,}"
)


criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

@torch.no_grad()
def evaluate(
    model,
    loader
):

    model.eval()

    all_true = []
    all_pred = []

    running_loss = 0.0

    for X, y in loader:

        X = X.to(device)
        y = y.to(device)

        logits = model(X)

        loss = criterion(
            logits,
            y
        )

        running_loss += (
            loss.item()
            * len(y)
        )

        pred = torch.argmax(
            logits,
            dim=1
        )

        all_true.extend(
            y.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_pred.extend(
            pred.detach()
            .cpu()
            .numpy()
            .tolist()
        )

    loss = (
        running_loss
        / len(loader.dataset)
    )

    macro_f1 = f1_score(
        all_true,
        all_pred,
        average="macro",
        zero_division=0
    )

    accuracy = accuracy_score(
        all_true,
        all_pred
    )

    return (
        loss,
        macro_f1,
        accuracy,
        np.asarray(all_true),
        np.asarray(all_pred)
    )


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

best_f1 = -1.0
best_state = None
patience_counter = 0

history = []


for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    model.train()

    running_loss = 0.0

    for X, y in train_loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(X)

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * len(y)
        )


    train_loss = (
        running_loss
        / len(train_loader.dataset)
    )


    (
        val_loss,
        val_f1,
        val_accuracy,
        _,
        _
    ) = evaluate(
        model,
        val_loader
    )


    print(
        f"Epoch {epoch:02d} | "
        f"train loss {train_loss:.4f} | "
        f"val loss {val_loss:.4f} | "
        f"val macro-F1 {val_f1:.4f} | "
        f"val acc {val_accuracy:.4f}"
    )


    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_macro_f1": val_f1,
        "val_accuracy": val_accuracy
    })


    # Save according to macro-F1,
    # not raw accuracy.
    if val_f1 > best_f1:

        best_f1 = val_f1

        best_state = copy.deepcopy(
            {
                k:
                v.detach()
                .cpu()
                .clone()

                for k, v
                in model.state_dict().items()
            }
        )

        patience_counter = 0

        print(
            "  ✅ New best model"
        )

    else:

        patience_counter += 1


    if patience_counter >= PATIENCE:

        print(
            "\nEarly stopping."
        )

        break


# --------------------------------------------------
# RESTORE BEST MODEL
# --------------------------------------------------

model.load_state_dict(
    best_state
)

model = model.to(device)


# --------------------------------------------------
# TEST SET
# --------------------------------------------------

(
    test_loss,
    test_f1,
    test_accuracy,
    y_true,
    y_pred
) = evaluate(
    model,
    test_loader
)


print("\n" + "=" * 70)

print("TEST RESULTS")

print("=" * 70)

print(
    f"Test loss:     {test_loss:.4f}"
)

print(
    f"Test accuracy: {test_accuracy:.4f}"
)

print(
    f"Test macro-F1: {test_f1:.4f}"
)


print("\nCLASSIFICATION REPORT\n")

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(
            range(NUM_CLASSES)
        ),
        target_names=LABELS,
        digits=3,
        zero_division=0
    )
)


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(
        range(NUM_CLASSES)
    )
)

cm_df = pd.DataFrame(
    cm,
    index=LABELS,
    columns=LABELS
)

cm_df.to_csv(
    os.path.join(
        MODEL_DIR,
        "confusion_matrix.csv"
    )
)


fig, ax = plt.subplots(
    figsize=(8, 7)
)

image = ax.imshow(cm)

ax.set_xticks(
    range(NUM_CLASSES)
)

ax.set_yticks(
    range(NUM_CLASSES)
)

ax.set_xticklabels(
    LABELS,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(
    LABELS
)

ax.set_xlabel(
    "Predicted"
)

ax.set_ylabel(
    "Actual"
)

ax.set_title(
    "ChildPlay Hand Movement Confusion Matrix"
)


for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


fig.tight_layout()

fig.savefig(
    os.path.join(
        MODEL_DIR,
        "confusion_matrix.png"
    ),
    dpi=180
)

plt.close(fig)


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

checkpoint = {

    "model_state_dict":
        best_state,

    "feature_mean":
        feature_mean,

    "feature_std":
        feature_std,

    "labels":
        LABELS,

    "sequence_length":
        20,

    "feature_dimension":
        63,

    "best_val_macro_f1":
        best_f1,

    "test_macro_f1":
        test_f1,

    "test_accuracy":
        test_accuracy
}


torch.save(
    checkpoint,
    os.path.join(
        MODEL_DIR,
        "child_hand_temporal.pt"
    )
)


pd.DataFrame(
    history
).to_csv(
    os.path.join(
        MODEL_DIR,
        "training_history.csv"
    ),
    index=False
)


print("\nSaved:")

print(
    "mvp/models/child_hand_temporal.pt"
)

print(
    "mvp/models/confusion_matrix.png"
)

print(
    "mvp/models/confusion_matrix.csv"
)

print(
    "mvp/models/training_history.csv"
)
