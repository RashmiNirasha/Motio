import json
import os
import sys
import threading
import time
import uuid

import numpy as np
import torch

from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool
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
from model_pincer import PincerTemporalNet
from observe.hand_state import HandReading, ObserveConfig
from observe.session import SessionStore
from pincer_features import (
    LABELS as PINCER_LABELS,
    SEQUENCE_LENGTH as PINCER_SEQUENCE_LENGTH,
    heuristic_pincer_probability,
    landmarks_to_features
)


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

LIVE_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "mvp",
    "models_live",
    "child_interaction_live.pt"
)

PINCER_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "mvp",
    "models_pincer",
    "pincer_model.pt"
)

PINCER_DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "mvp",
    "pincer_data",
    "raw"
)

OBSERVE_LOG_DIR = os.path.join(
    PROJECT_ROOT,
    "mvp",
    "observe_logs"
)

CANDIDATE_DIR = os.path.join(
    PROJECT_ROOT,
    "mvp",
    "pincer_candidates"
)

CANDIDATE_LABELS_PATH = os.path.join(
    CANDIDATE_DIR,
    "labels.json"
)

VIDEO_AGE_PATH = os.path.join(
    CANDIDATE_DIR,
    "video_age.json"
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


# --------------------------------------------------
# Live-compatible interaction model (shape only)
#
# The original model relies on ChildPlay's per-joint confidence
# and the hand's position/size in the frame, which MediaPipe
# doesn't provide, so live it outputs ~0.5 regardless of the hand.
# train_live_interaction.py trains the same architecture on hand
# shape only. It is used whenever it exists, unless
# MOTIO_INTERACTION_MODEL=original.
# --------------------------------------------------

interaction_model_name = "child_object_interaction"
masked_columns = None
masked_values = None

if (
    os.path.exists(LIVE_MODEL_PATH)
    and os.environ.get("MOTIO_INTERACTION_MODEL", "live") != "original"
):

    live_checkpoint = torch.load(
        LIVE_MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    model = BinaryTemporalHandNet(
        np.asarray(live_checkpoint["feature_mean"], dtype=np.float32),
        np.asarray(live_checkpoint["feature_std"], dtype=np.float32)
    )

    model.load_state_dict(live_checkpoint["model_state_dict"])
    model.eval()

    masked_columns = np.asarray(live_checkpoint["masked_columns"])
    masked_values = np.asarray(
        live_checkpoint["constants"], dtype=np.float32
    )[masked_columns]

    threshold = float(live_checkpoint["threshold"])
    interaction_model_name = "child_interaction_live (hand shape only)"


print("Model loaded:", interaction_model_name)
print("Threshold:", threshold)


# --------------------------------------------------
# Load pincer model (optional)
#
# Until train_pincer.py has been run, pincer
# predictions use the rule-based heuristic.
# --------------------------------------------------

pincer_model = None

# Heuristic scores are not calibrated probabilities;
# 0.5 is a reasonable starting cut-off.
pincer_threshold = 0.5

if os.path.exists(PINCER_MODEL_PATH):

    pincer_checkpoint = torch.load(
        PINCER_MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    pincer_model = PincerTemporalNet(
        pincer_checkpoint["feature_mean"],
        pincer_checkpoint["feature_std"]
    )

    pincer_model.load_state_dict(
        pincer_checkpoint["model_state_dict"]
    )

    pincer_model.eval()

    pincer_threshold = float(
        pincer_checkpoint["threshold"]
    )

    print("Pincer model loaded. Threshold:", pincer_threshold)

else:

    print(
        "No pincer model found, using rule-based pincer heuristic."
    )


def pincer_source():

    return "model" if pincer_model is not None else "heuristic"


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


class HandPredictionRequest(BaseModel):
    # [20, 69] features for the interaction model
    sequence: list[list[float]]
    # [20, 21, 3] landmarks in isotropic pixel units
    landmarks: list[list[list[float]]]


class CandidateLabel(BaseModel):
    candidate_id: str
    # pincer / other_grasp / no_grasp / unclear
    label: str
    # One entry per crop frame: [21, 3] landmarks in
    # crop pixels, or null where MediaPipe found no hand.
    landmarks: list[list[list[float]] | None]


class VideoAge(BaseModel):
    video_id: str
    # under5 / 5plus / unsure
    age_band: str


class PincerRecording(BaseModel):
    label: str
    session_id: str
    source: str
    source_name: str = ""
    handedness: str = ""
    landmarks: list[list[list[float]]]


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
        "model": interaction_model_name,
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
            joint_confidence_fill,
        "pincer_threshold": pincer_threshold,
        "pincer_source": pincer_source()
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

    probability = interaction_probability(array)

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


def interaction_probability(array):

    if masked_columns is not None:

        # Columns the live model never saw during training.
        array = array.copy()
        array[:, masked_columns] = masked_values

    tensor = torch.from_numpy(
        array
    ).unsqueeze(0)

    with torch.no_grad():

        return torch.sigmoid(
            model(tensor)
        ).item()


def pincer_probability(landmarks):

    if pincer_model is None:

        return heuristic_pincer_probability(
            landmarks
        )

    features = torch.from_numpy(
        landmarks_to_features(landmarks)
    ).unsqueeze(0)

    with torch.no_grad():

        return torch.sigmoid(
            pincer_model(features)
        ).item()


@app.post("/predict_hand")
def predict_hand(
    request: HandPredictionRequest
):

    # One request per tracked hand:
    # object interaction + pincer grasp.

    sequence = np.asarray(
        request.sequence,
        dtype=np.float32
    )

    landmarks = np.asarray(
        request.landmarks,
        dtype=np.float32
    )

    if sequence.shape != (20, 69):

        raise HTTPException(
            status_code=400,
            detail=(
                "Expected sequence [20,69], "
                f"received {list(sequence.shape)}"
            )
        )

    if landmarks.shape != (PINCER_SEQUENCE_LENGTH, 21, 3):

        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected landmarks [{PINCER_SEQUENCE_LENGTH},21,3], "
                f"received {list(landmarks.shape)}"
            )
        )

    interaction_p = interaction_probability(sequence)
    pincer_p = pincer_probability(landmarks)

    return {
        "interaction_probability":
            round(float(interaction_p), 6),

        "interaction":
            bool(interaction_p >= threshold),

        "threshold":
            threshold,

        "pincer_probability":
            round(float(pincer_p), 6),

        "pincer":
            bool(pincer_p >= pincer_threshold),

        "pincer_threshold":
            pincer_threshold,

        "pincer_source":
            pincer_source()
    }


# --------------------------------------------------
# Observation MVP (Stage 1: hands)
#
# The browser sends the current 20-frame buffers for each visible
# hand a few times per second. The server scores each hand with the
# ChildPlay interaction model and keeps the per-hand temporal state
# (smoothing, confirmed interaction, inferred phase).
# --------------------------------------------------

observe_sessions = SessionStore(
    config=ObserveConfig(threshold=threshold),
    log_dir=OBSERVE_LOG_DIR
)


class ObserveHand(BaseModel):
    slot: int
    handedness: str = "Unknown"
    # [20, 69] features and [20, 21, 3] pixel landmarks
    sequence: list[list[float]]
    landmarks: list[list[list[float]]]


class ObserveUpdate(BaseModel):
    session_id: str
    # Client clock in seconds (only differences matter)
    t: float
    hands: list[ObserveHand] = []
    # Video width, height in pixels (the landmarks' coordinate frame)
    video_size: list[float] | None = None


class ObserveSessionRequest(BaseModel):
    session_id: str
    # record endpoint only
    on: bool = True


@app.post("/observe/update")
def observe_update(
    request: ObserveUpdate
):

    readings = []

    for hand in request.hands:

        sequence = np.asarray(hand.sequence, dtype=np.float32)
        landmarks = np.asarray(hand.landmarks, dtype=np.float32)

        if sequence.shape != (20, 69) or landmarks.shape != (20, 21, 3):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Expected sequence [20,69] and landmarks [20,21,3], "
                    f"received {list(sequence.shape)} and {list(landmarks.shape)}"
                )
            )

        wrist = landmarks[-1, 0]
        palm = float(np.linalg.norm(landmarks[-1, 0] - landmarks[-1, 9]))

        readings.append(
            HandReading(
                slot=hand.slot,
                p=float(interaction_probability(sequence)),
                wrist=(float(wrist[0]), float(wrist[1])),
                palm=max(palm, 1e-6),
                handedness=hand.handedness
            )
        )

    session = observe_sessions.get(request.session_id)

    result = session.update(
        request.t,
        readings,
        hand_frames={
            hand.slot: hand.landmarks[-1]
            for hand in request.hands
        },
        video_size=request.video_size
    )

    result["threshold"] = threshold
    result["recording"] = session.recording

    return result


# --------------------------------------------------
# Observation MVP (Stage 2: play objects)
#
# The browser sends a small JPEG of the current frame about twice a
# second. It is decoded in memory, run through the detector and
# discarded; frames are never written to disk. The detector loads on
# the first frame (a few seconds), so the server still starts fast.
# --------------------------------------------------

object_detector = None
object_detector_error = None
object_detector_lock = threading.Lock()


def get_object_detector():

    global object_detector, object_detector_error

    with object_detector_lock:

        if object_detector is None and object_detector_error is None:

            try:
                from observe.detector import ObjectDetector

                detector = ObjectDetector()

                # The first inference takes seconds (GPU start-up) and each new
                # image shape ~0.7 s more, so warm up the common camera shapes
                # (4:3, 16:9, portrait) here instead of on the first frames.
                for height, width in [(480, 640), (360, 640), (640, 360)]:
                    detector.detect(np.zeros((height, width, 3), dtype=np.uint8))

                object_detector = detector
                print("Object detector ready on", detector.device, flush=True)

            except Exception as error:

                object_detector_error = f"{type(error).__name__}: {error}"
                print("Object detector unavailable:", object_detector_error, flush=True)

    return object_detector


# Load in the background at start-up, so it is ready before the first frame.
# Set MOTIO_DETECTOR=off to skip (the server then works without objects).
if os.environ.get("MOTIO_DETECTOR", "on") != "off":
    threading.Thread(target=get_object_detector, daemon=True).start()


@app.post("/observe/frame")
async def observe_frame(
    request: Request,
    session_id: str,
    t: float
):

    import cv2

    data = await request.body()

    image = cv2.imdecode(
        np.frombuffer(data, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(status_code=400, detail="Body must be a JPEG or PNG image")

    detector = await run_in_threadpool(get_object_detector)

    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Object detector unavailable: " + str(object_detector_error)
        )

    detections = await run_in_threadpool(detector.detect, image)

    objects = [d.to_dict() for d in detections]

    observe_sessions.get(session_id).last_objects = {"t": t, "objects": objects}

    return {
        "t": t,
        "objects": objects,
        "ms": round(detector.last_ms),
        "device": detector.device,
        "image_size": [int(image.shape[1]), int(image.shape[0])]
    }


# --------------------------------------------------
# Caregiver assistant (Claude API)
# --------------------------------------------------

class ProfileRequest(BaseModel):
    session_id: str
    child_age_months: int | None = None


class AskRequest(BaseModel):
    session_id: str
    question: str


caregiver_assistant = None


@app.post("/observe/profile")
def observe_profile(
    request: ProfileRequest
):

    if request.child_age_months is not None and not 0 <= request.child_age_months <= 72:
        raise HTTPException(status_code=400, detail="Age must be 0-72 months")

    observe_sessions.get(request.session_id).child_age_months = request.child_age_months

    return {"child_age_months": request.child_age_months}


@app.get("/observe/observation")
def observe_observation(
    session_id: str
):

    # Exactly what the assistant would receive right now.
    return observe_sessions.get(session_id).observation()


@app.post("/assistant/ask")
def assistant_ask(
    request: AskRequest
):

    global caregiver_assistant

    from observe.assistant import Assistant, AssistantUnavailable

    question = request.question.strip()[:1000]

    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    if caregiver_assistant is None:
        caregiver_assistant = Assistant()

    session = observe_sessions.get(request.session_id)
    observation = session.observation()

    try:
        reply = caregiver_assistant.respond(observation, question, session.chat)
    except AssistantUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error))

    reply["observation"] = observation

    return reply


@app.post("/assistant/reset")
def assistant_reset(
    request: ObserveSessionRequest
):

    observe_sessions.get(request.session_id).chat = []

    return {"reset": True}


@app.post("/observe/reset")
def observe_reset(
    request: ObserveSessionRequest
):

    observe_sessions.get(request.session_id).reset()

    return {"reset": True}


@app.post("/observe/record")
def observe_record(
    request: ObserveSessionRequest
):

    session = observe_sessions.get(request.session_id)

    if request.on:
        path = session.start_recording()
    else:
        path = session.log_path
        session.stop_recording()

    return {
        "recording": session.recording,
        "file": os.path.relpath(path, PROJECT_ROOT) if path else None
    }


# --------------------------------------------------
# Pincer data collection
# --------------------------------------------------

@app.post("/record_pincer")
def record_pincer(
    recording: PincerRecording
):

    if recording.label not in PINCER_LABELS:

        raise HTTPException(
            status_code=400,
            detail=f"label must be one of {PINCER_LABELS}"
        )

    landmarks = np.asarray(
        recording.landmarks,
        dtype=np.float32
    )

    if (
        landmarks.ndim != 3
        or landmarks.shape[1:] != (21, 3)
        or len(landmarks) < PINCER_SEQUENCE_LENGTH
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected at least {PINCER_SEQUENCE_LENGTH} frames "
                f"of [21,3] landmarks, received {list(landmarks.shape)}"
            )
        )

    os.makedirs(
        PINCER_DATA_DIR,
        exist_ok=True
    )

    clip_id = (
        time.strftime("%Y%m%d-%H%M%S")
        + "_"
        + recording.label
        + "_"
        + uuid.uuid4().hex[:8]
    )

    with open(
        os.path.join(PINCER_DATA_DIR, clip_id + ".json"),
        "w"
    ) as f:

        json.dump(
            {
                **recording.model_dump(),
                "clip_id": clip_id,
                "created": time.time()
            },
            f
        )

    return {
        "saved": clip_id,
        "frames": len(landmarks)
    }


@app.get("/pincer_data_stats")
def pincer_data_stats():

    stats = {
        label: {"clips": 0, "frames": 0}
        for label in PINCER_LABELS
    }

    sessions = set()

    if os.path.isdir(PINCER_DATA_DIR):

        for name in os.listdir(PINCER_DATA_DIR):

            if not name.endswith(".json"):
                continue

            with open(os.path.join(PINCER_DATA_DIR, name)) as f:
                clip = json.load(f)

            if clip["label"] in stats:
                stats[clip["label"]]["clips"] += 1
                stats[clip["label"]]["frames"] += len(clip["landmarks"])

            sessions.add(clip["session_id"])

    return {
        "labels": stats,
        "sessions": len(sessions),
        "pincer_source": pincer_source()
    }


# --------------------------------------------------
# ChildPlay candidate labelling
#
# Candidates come from extract_pincer_candidates.py.
# Each label is stored in labels.json, and labelled
# clips (except "unclear") are exported into the same
# folder as recorder clips so train_pincer.py uses them.
# --------------------------------------------------

CANDIDATE_LABEL_CHOICES = PINCER_LABELS + ["unclear"]

AGE_BANDS = ["under5", "5plus", "unsure"]


def load_candidates():

    path = os.path.join(CANDIDATE_DIR, "candidates.json")

    if not os.path.exists(path):
        return {}

    with open(path) as f:
        return {c["candidate_id"]: c for c in json.load(f)}


def load_candidate_labels():

    if not os.path.exists(CANDIDATE_LABELS_PATH):
        return {}

    with open(CANDIDATE_LABELS_PATH) as f:
        return json.load(f)


def load_video_ages():

    # Age band per source video, confirmed by the labeller
    # (ChildPlay does not annotate age).
    if not os.path.exists(VIDEO_AGE_PATH):
        return {}

    with open(VIDEO_AGE_PATH) as f:
        return json.load(f)


def export_candidate(candidate, label, landmarks):

    os.makedirs(PINCER_DATA_DIR, exist_ok=True)

    prefix = "childplay_" + candidate["candidate_id"] + "_"

    # Relabelling replaces any earlier export.
    for name in os.listdir(PINCER_DATA_DIR):
        if name.startswith(prefix):
            os.remove(os.path.join(PINCER_DATA_DIR, name))

    if label == "unclear":
        return 0

    # Only continuous runs with a MediaPipe hand
    # in every frame become training clips.
    runs = []
    current = []

    for frame in landmarks + [None]:

        if frame is not None:
            current.append(frame)
            continue

        if len(current) >= PINCER_SEQUENCE_LENGTH:
            runs.append(current)

        current = []

    for i, run in enumerate(runs):

        with open(os.path.join(PINCER_DATA_DIR, f"{prefix}{i}.json"), "w") as f:

            json.dump(
                {
                    "label": label,
                    # Split by source video, so one video never
                    # appears in both training and evaluation.
                    "session_id": "childplay_" + candidate["video_id"],
                    "source": "childplay",
                    "source_name": candidate["candidate_id"],
                    "handedness": candidate["hand"],
                    "is_child": bool(candidate.get("is_child")),
                    "landmarks": run,
                    "clip_id": f"{prefix}{i}",
                    "created": time.time()
                },
                f
            )

    return len(runs)


@app.get("/pincer_candidates")
def pincer_candidates():

    labels = load_candidate_labels()
    ages = load_video_ages()

    return [
        {
            "candidate_id": cid,
            "video_id": c["video_id"],
            "video_title": c.get("video_title", ""),
            "age_hint": c.get("age_hint", "unknown"),
            "age_reasons": c.get("age_reasons", []),
            "age_band": ages.get(c["video_id"]),
            "hand": c["hand"],
            "n_frames": c["n_frames"],
            "actions": c["actions"],
            **labels.get(cid, {})
        }
        for cid, c in load_candidates().items()
    ]


@app.post("/video_age")
def video_age(
    request: VideoAge
):

    if request.age_band not in AGE_BANDS:
        raise HTTPException(
            status_code=400,
            detail=f"age_band must be one of {AGE_BANDS}"
        )

    ages = load_video_ages()
    ages[request.video_id] = request.age_band

    with open(VIDEO_AGE_PATH, "w") as f:
        json.dump(ages, f, indent=1)

    return {"video_id": request.video_id, "age_band": request.age_band}


@app.post("/pincer_label")
def pincer_label(
    request: CandidateLabel
):

    candidate = load_candidates().get(request.candidate_id)

    if candidate is None:
        raise HTTPException(status_code=404, detail="Unknown candidate")

    if request.label not in CANDIDATE_LABEL_CHOICES:
        raise HTTPException(
            status_code=400,
            detail=f"label must be one of {CANDIDATE_LABEL_CHOICES}"
        )

    if len(request.landmarks) != candidate["n_frames"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected {candidate['n_frames']} landmark frames, "
                f"received {len(request.landmarks)}"
            )
        )

    exported = export_candidate(
        candidate,
        request.label,
        request.landmarks
    )

    labels = load_candidate_labels()

    labels[request.candidate_id] = {
        "label": request.label,
        # Where the clip is, so re-running the extractor with
        # different settings never re-offers the same footage.
        "clip": candidate["clip"],
        "pid": candidate["pid"],
        "hand": candidate["hand"],
        "start_frame": candidate["start_frame"],
        "end_frame": candidate["end_frame"],
        "hand_frames": sum(f is not None for f in request.landmarks),
        "exported_clips": exported,
        "labelled_at": time.time()
    }

    with open(CANDIDATE_LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=1)

    return labels[request.candidate_id]


# --------------------------------------------------
# Frontend
# --------------------------------------------------

print(
    "Serving frontend from:",
    STATIC_DIR
)

os.makedirs(
    CANDIDATE_DIR,
    exist_ok=True
)


class NoCacheStaticFiles(StaticFiles):

    # The pages and hand-tracking.js change together. Without this,
    # a browser can pair a new page with a cached old script and the
    # whole page silently stops working. "no-cache" still lets the
    # browser reuse a file, but only after the server confirms it
    # hasn't changed (ETag), so it is cheap.

    async def get_response(self, path, scope):

        response = await super().get_response(path, scope)

        response.headers["Cache-Control"] = "no-cache"

        return response

app.mount(
    "/candidates",
    StaticFiles(
        directory=CANDIDATE_DIR
    ),
    name="candidates"
)

app.mount(
    "/app",
    NoCacheStaticFiles(
        directory=STATIC_DIR,
        html=True
    ),
    name="app"
)
