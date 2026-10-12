import os
import sys

import numpy as np
import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.insert(
    0,
    os.path.join(
        PROJECT_ROOT,
        "mvp"
    )
)

from model_binary import BinaryTemporalHandNet


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "mvp",
    "models_binary",
    "child_object_interaction.pt"
)

THRESHOLD_PATH = os.path.join(
    PROJECT_ROOT,
    "mvp",
    "models_binary",
    "iphone_threshold.txt"
)

STATIC_DIR = os.path.join(
    os.path.dirname(__file__),
    "static"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("Loading model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

feature_mean = np.asarray(
    checkpoint["feature_mean"],
    dtype=np.float32
)

feature_std = np.asarray(
    checkpoint["feature_std"],
    dtype=np.float32
)

model = BinaryTemporalHandNet(
    feature_mean,
    feature_std
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


with open(
    THRESHOLD_PATH,
    "r"
) as f:
    threshold = float(
        f.read().strip()
    )


# ChildPlay used:
# [x, y, confidence] × 21 = first 63 features
#
# MediaPipe Web does not provide the same
# confidence values, so use training means.
joint_confidence_fill = (
    feature_mean[:63]
    .reshape(21, 3)[:, 2]
    .tolist()
)


print("Model loaded.")
print("Threshold:", threshold)


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="ChildPlay MVP API",
    version="0.2"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


class PredictionRequest(BaseModel):
    sequence: list[list[float]]


# --------------------------------------------------
# API
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "service": "ChildPlay MVP",
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "ok",
        "model": "child_object_interaction",
        "sequence_length": 20,
        "feature_dimension": 69,
        "threshold": threshold
    }


@app.get("/config")
def config():

    return {
        "sequence_length": 20,
        "feature_dimension": 69,
        "threshold": threshold,
        "confirmation_required": 3,
        "confirmation_window": 4,
        "joint_confidence_fill":
            joint_confidence_fill
    }


@app.post("/predict")
def predict(
    request: PredictionRequest
):

    array = np.asarray(
        request.sequence,
        dtype=np.float32
    )

    if array.shape != (20, 69):

        raise HTTPException(
            status_code=400,
            detail=(
                "Expected [20,69], "
                f"received {list(array.shape)}"
            )
        )

    tensor = torch.from_numpy(
        array
    ).unsqueeze(0)

    with torch.no_grad():

        logit = model(tensor)

        probability = (
            torch.sigmoid(logit)
            .item()
        )

    interaction = (
        probability >= threshold
    )

    return {
        "interaction_probability":
            round(float(probability), 6),

        "threshold":
            threshold,

        "interaction":
            bool(interaction),

        "label":
            (
                "object_interaction"
                if interaction
                else "no_interaction"
            )
    }


# --------------------------------------------------
# Frontend
# --------------------------------------------------

print(
    "Serving frontend from:",
    STATIC_DIR
)

app.mount(
    "/app",
    StaticFiles(
        directory=STATIC_DIR,
        html=True
    ),
    name="app"
)
