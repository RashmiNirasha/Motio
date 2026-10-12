import os
import sys
import copy

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

sys.path.append("mvp")

from model_binary import BinaryTemporalHandNet


DATA_DIR = "mvp/dataset_v2"
MODEL_DIR = "mvp/models_binary"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

BATCH_SIZE = 256
MAX_EPOCHS = 35
PATIENCE = 7
LEARNING_RATE = 1e-3

np.random.seed(42)
torch.manual_seed(42)


# -----------------------------------
# DEVICE
# -----------------------------------

if torch.backends.mps.is_available():

    device = torch.device("mps")

elif torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print("\nUsing device:", device)


# -----------------------------------
# LOAD
# -----------------------------------

def load_split(split):

    X = np.load(
        f"{DATA_DIR}/X_{split}.npy"
    ).astype(np.float32)

    original_y = np.load(
        f"{DATA_DIR}/y_{split}.npy"
    ).astype(np.int64)

    # V2:
    # 0 = background
    # 1 = object_interaction
    # 2 = point
    #
    # Binary:
    # object_interaction = 1
    # everything else = 0

    y = (
        original_y == 1
    ).astype(np.float32)

    return X, y


X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test")


print("\nDataset:")

print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)


def print_counts(name, y):

    positive = int(y.sum())
    negative = int(len(y) - positive)

    print(
        f"{name}: "
        f"NO={negative}, "
        f"OBJECT={positive}"
    )


print_counts("TRAIN", y_train)
print_counts("VAL", y_val)
print_counts("TEST", y_test)


# -----------------------------------
# NORMALIZATION
# -----------------------------------

feature_mean = X_train.mean(
    axis=(0, 1)
)

feature_std = X_train.std(
    axis=(0, 1)
)

feature_std[
    feature_std < 1e-6
] = 1.0


np.savez(
    f"{MODEL_DIR}/normalization.npz",
    mean=feature_mean,
    std=feature_std
)


# -----------------------------------
# LOADERS
# -----------------------------------

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


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
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


# -----------------------------------
# MODEL
# -----------------------------------

model = BinaryTemporalHandNet(
    feature_mean,
    feature_std
).to(device)


print(
    "\nParameters:",
    f"{sum(p.numel() for p in model.parameters()):,}"
)


# Mild positive weighting
negative_count = (
    len(y_train) - y_train.sum()
)

positive_count = y_train.sum()

pos_weight = torch.tensor(
    [
        negative_count
        /
        positive_count
    ],
    dtype=torch.float32,
    device=device
)


print(
    "Positive weight:",
    pos_weight.item()
)


criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# -----------------------------------
# PREDICTION
# -----------------------------------

@torch.no_grad()
def predict_scores(
    model,
    loader
):

    model.eval()

    true = []
    scores = []

    total_loss = 0.0

    for X, y in loader:

        X = X.to(device)
        y = y.to(device)

        logits = model(X)

        loss = criterion(
            logits,
            y
        )

        total_loss += (
            loss.item()
            * len(y)
        )

        probabilities = torch.sigmoid(
            logits
        )

        true.extend(
            y.cpu()
            .numpy()
            .tolist()
        )

        scores.extend(
            probabilities.cpu()
            .numpy()
            .tolist()
        )

    return (
        total_loss / len(loader.dataset),
        np.asarray(true),
        np.asarray(scores)
    )


def find_best_threshold(
    true,
    scores
):

    best_threshold = 0.5
    best_f1 = -1

    for threshold in np.arange(
        0.20,
        0.91,
        0.01
    ):

        pred = (
            scores >= threshold
        ).astype(int)

        score = f1_score(
            true,
            pred,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold

    return (
        float(best_threshold),
        float(best_f1)
    )


# -----------------------------------
# TRAIN
# -----------------------------------

best_val_f1 = -1
best_state = None
best_threshold = 0.5

patience_counter = 0

history = []


for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    model.train()

    total_loss = 0.0


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

        total_loss += (
            loss.item()
            * len(y)
        )


    train_loss = (
        total_loss
        /
        len(train_loader.dataset)
    )


    val_loss, val_true, val_scores = (
        predict_scores(
            model,
            val_loader
        )
    )


    threshold, val_f1 = (
        find_best_threshold(
            val_true,
            val_scores
        )
    )


    val_pred = (
        val_scores >= threshold
    ).astype(int)


    val_precision = precision_score(
        val_true,
        val_pred,
        zero_division=0
    )

    val_recall = recall_score(
        val_true,
        val_pred,
        zero_division=0
    )


    print(
        f"Epoch {epoch:02d} | "
        f"train {train_loss:.4f} | "
        f"val {val_loss:.4f} | "
        f"F1 {val_f1:.4f} | "
        f"P {val_precision:.4f} | "
        f"R {val_recall:.4f} | "
        f"threshold {threshold:.2f}"
    )


    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_f1": val_f1,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "threshold": threshold
    })


    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        best_threshold = threshold

        best_state = copy.deepcopy({
            k:
            v.detach()
            .cpu()
            .clone()

            for k, v
            in model.state_dict().items()
        })

        patience_counter = 0

        print("  ✅ New best")

    else:

        patience_counter += 1


    if patience_counter >= PATIENCE:

        print("\nEarly stopping.")

        break


# -----------------------------------
# RESTORE BEST
# -----------------------------------

model.load_state_dict(
    best_state
)

model = model.to(device)


# Recalculate threshold with best model
_, val_true, val_scores = (
    predict_scores(
        model,
        val_loader
    )
)

best_threshold, _ = (
    find_best_threshold(
        val_true,
        val_scores
    )
)


# -----------------------------------
# TEST
# -----------------------------------

test_loss, test_true, test_scores = (
    predict_scores(
        model,
        test_loader
    )
)

test_pred = (
    test_scores >= best_threshold
).astype(int)


test_accuracy = accuracy_score(
    test_true,
    test_pred
)

test_precision = precision_score(
    test_true,
    test_pred,
    zero_division=0
)

test_recall = recall_score(
    test_true,
    test_pred,
    zero_division=0
)

test_f1 = f1_score(
    test_true,
    test_pred,
    zero_division=0
)


print("\n" + "=" * 70)
print("BINARY MVP TEST RESULTS")
print("=" * 70)

print(
    f"Threshold: {best_threshold:.2f}"
)

print(
    f"Accuracy:  {test_accuracy:.4f}"
)

print(
    f"Precision: {test_precision:.4f}"
)

print(
    f"Recall:    {test_recall:.4f}"
)

print(
    f"F1:        {test_f1:.4f}"
)


print("\nCLASSIFICATION REPORT\n")

print(
    classification_report(
        test_true,
        test_pred,
        target_names=[
            "no_interaction",
            "object_interaction"
        ],
        digits=3,
        zero_division=0
    )
)


# -----------------------------------
# CONFUSION MATRIX
# -----------------------------------

cm = confusion_matrix(
    test_true,
    test_pred
)

print("\nCONFUSION MATRIX")
print(cm)


pd.DataFrame(
    cm,
    index=[
        "actual_no",
        "actual_object"
    ],
    columns=[
        "pred_no",
        "pred_object"
    ]
).to_csv(
    f"{MODEL_DIR}/confusion_matrix.csv"
)


fig, ax = plt.subplots(
    figsize=(5, 5)
)

ax.imshow(cm)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels([
    "No interaction",
    "Object interaction"
])

ax.set_yticklabels([
    "No interaction",
    "Object interaction"
])

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

fig.tight_layout()

fig.savefig(
    f"{MODEL_DIR}/confusion_matrix.png",
    dpi=180
)

plt.close(fig)


# -----------------------------------
# SAVE
# -----------------------------------

checkpoint = {

    "model_state_dict":
        best_state,

    "feature_mean":
        feature_mean,

    "feature_std":
        feature_std,

    "sequence_length":
        20,

    "feature_dimension":
        69,

    "threshold":
        best_threshold,

    "test_accuracy":
        test_accuracy,

    "test_precision":
        test_precision,

    "test_recall":
        test_recall,

    "test_f1":
        test_f1
}


torch.save(
    checkpoint,
    f"{MODEL_DIR}/child_object_interaction.pt"
)


pd.DataFrame(
    history
).to_csv(
    f"{MODEL_DIR}/training_history.csv",
    index=False
)


with open(
    f"{MODEL_DIR}/threshold.txt",
    "w"
) as f:

    f.write(
        str(best_threshold)
    )


print("\nSaved:")
print(
    "mvp/models_binary/child_object_interaction.pt"
)
print(
    "mvp/models_binary/confusion_matrix.png"
)
print(
    "mvp/models_binary/threshold.txt"
)
