import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from model_binary import BinaryTemporalHandNet


# --------------------------------------------------
# Train an interaction model that can run on MediaPipe.
#
# The original model (train_binary_mvp.py) turned out not to use
# the shape of the hand: its ChildPlay accuracy comes from the pose
# estimator's per-joint confidence and from the hand's position and
# size in the frame, which MediaPipe does not provide (see the
# ablation in the project notes / README).
#
# This variant keeps the same architecture and the same 69-column
# input, so the server and web page are unchanged, but during training
# the confidence and position/size columns are replaced by constants.
# It can only learn from hand shape (plus its change over time).
# It is also trained with mirroring, rotation and scale jitter, because
# MediaPipe's landmarks differ in small ways from ChildPlay's.
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "mvp", "dataset_v2"))
parser.add_argument("--model-dir", default=os.path.join(PROJECT_ROOT, "mvp", "models_live"))
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--no-save", action="store_true")
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)

# Column groups of the 69-feature frame
CONF = np.arange(2, 63, 3)
XCOL = np.arange(0, 63, 3)
YCOL = np.arange(1, 63, 3)
GLOBAL = np.arange(63, 69)
MASKED = np.concatenate([CONF, GLOBAL])


def load(split):

    X = np.load(os.path.join(args.data_dir, f"X_{split}.npy")).astype(np.float32)
    y = (np.load(os.path.join(args.data_dir, f"y_{split}.npy")) == 1).astype(np.float32)

    return X, y


X_train, y_train = load("train")
X_val, y_val = load("val")
X_test, y_test = load("test")

# Constants for the masked columns (training means)
mean = X_train.reshape(-1, 69).mean(axis=0)
std = X_train.reshape(-1, 69).std(axis=0) + 1e-6

# Masked columns become exactly constant: mean = that constant, std = 1.
std[MASKED] = 1.0

constants = mean.copy()


def mask(X):

    X = X.copy()
    X[:, :, MASKED] = constants[MASKED]

    return X


X_train, X_val, X_test = mask(X_train), mask(X_val), mask(X_test)


def augment(X):

    X = X.copy()
    n = len(X)

    # Mirror: handedness labels are unreliable, so make the model
    # indifferent to left/right.
    flip = rng.random(n) < 0.5
    X[np.ix_(flip, np.arange(X.shape[1]), XCOL)] *= -1

    # Small in-plane rotation of the whole hand shape.
    angle = rng.uniform(-np.pi / 7, np.pi / 7, size=n)[:, None, None]
    x, y = X[:, :, XCOL].copy(), X[:, :, YCOL].copy()
    X[:, :, XCOL] = np.cos(angle) * x - np.sin(angle) * y
    X[:, :, YCOL] = np.sin(angle) * x + np.cos(angle) * y

    # Scale jitter: ChildPlay normalises by a padded detector box,
    # MediaPipe by the landmark extent.
    X[:, :, XCOL] *= rng.uniform(0.75, 1.3, size=(n, 1, 1))
    X[:, :, YCOL] *= rng.uniform(0.9, 1.1, size=(n, 1, 1))

    return X


def auc(scores, labels):

    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    positives = labels.sum()
    negatives = len(labels) - positives

    return (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)


def metrics(scores, labels, threshold):

    predicted = scores >= threshold
    actual = labels == 1

    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    tn = int(np.sum(~predicted & ~actual))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    }


model = BinaryTemporalHandNet(mean, std)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

pos_weight = torch.tensor((1 - y_train.mean()) / y_train.mean(), dtype=torch.float32)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def predict(X):

    model.eval()

    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(X))).numpy()


best_auc = -1.0
best_state = None

print(f"train {len(y_train)}  val {len(y_val)}  test {len(y_test)}  (positives {y_train.mean():.2f})")

for epoch in range(1, args.epochs + 1):

    model.train()

    order = rng.permutation(len(X_train))
    X_aug = augment(X_train)

    total = 0.0

    for start in range(0, len(order), args.batch_size):

        batch = order[start:start + args.batch_size]

        optimizer.zero_grad()

        loss = loss_fn(
            model(torch.from_numpy(X_aug[batch])),
            torch.from_numpy(y_train[batch])
        )

        loss.backward()
        optimizer.step()

        total += loss.item() * len(batch)

    val_auc = auc(predict(X_val), y_val)

    if val_auc > best_auc:
        best_auc = val_auc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch == 1 or epoch % 5 == 0:
        print(f"epoch {epoch:3d}  loss {total / len(order):.4f}  val AUC {val_auc:.3f}")

model.load_state_dict(best_state)

val_scores = predict(X_val)

# Threshold: best F1 on validation
best = max(
    (metrics(val_scores, y_val, t)["f1"], t)
    for t in np.arange(0.05, 0.96, 0.01)
)

threshold = float(round(best[1], 2))

test_scores = predict(X_test)
result = metrics(test_scores, y_test, threshold)

print(f"\nbest val AUC {best_auc:.3f}   threshold (val F1) {threshold:.2f}")
print(f"TEST AUC {auc(test_scores, y_test):.3f}")
print(
    f"TEST at {threshold:.2f}: accuracy {result['accuracy']:.3f}  "
    f"precision {result['precision']:.3f}  recall {result['recall']:.3f}  F1 {result['f1']:.3f}"
)
print("(original model for comparison: AUC 0.767; at 0.66 accuracy 0.719, precision 0.642, recall 0.441)")

# Robustness check: does the score move when the hand shape is scrambled?
shuffled = X_test.copy()
shuffled[:, :, np.concatenate([XCOL, YCOL])] = X_test[rng.permutation(len(X_test))][:, :, np.concatenate([XCOL, YCOL])]
print(f"TEST AUC with hand shape scrambled (should be ~0.5): {auc(predict(shuffled), y_test):.3f}")

if not args.no_save:

    os.makedirs(args.model_dir, exist_ok=True)

    path = os.path.join(args.model_dir, "child_interaction_live.pt")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "constants": constants,
            "masked_columns": MASKED.tolist(),
            "sequence_length": 20,
            "feature_dimension": 69,
            "threshold": threshold,
            "test_auc": float(auc(test_scores, y_test)),
            "test_accuracy": result["accuracy"],
            "test_precision": result["precision"],
            "test_recall": result["recall"],
            "test_f1": result["f1"]
        },
        path
    )

    print("saved", path)
