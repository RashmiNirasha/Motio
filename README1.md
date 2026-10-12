# ChildPlay Hand-Object Interaction MVP

A research prototype for **real-time child hand-object interaction monitoring** using the ChildPlay-Hand dataset, MediaPipe Hand Landmarker, a lightweight temporal PyTorch model, FastAPI, and a browser-based camera interface.

> **Research prototype only.** This system is not a medical device and should not be used to diagnose motor-development conditions or make clinical decisions.

## Overview

The current MVP detects whether a visible hand is likely engaged in an **object interaction**.

```text
Phone / laptop camera
        ↓
MediaPipe Hand Landmarker
        ↓
21 hand landmarks
        ↓
69 features per frame
        ↓
20-frame temporal buffer
        ↓
FastAPI backend
        ↓
PyTorch temporal classifier
        ↓
Object-interaction probability
        ↓
MONITORING / OBJECT INTERACTION
```

The current binary classifier uses these labels:

- `0` — no object interaction
- `1` — object interaction

For the MVP, ChildPlay actions such as `grasp`, `hold`, `release`, `operate`, `put`, `give`, `drop`, and `throw` are grouped into `object_interaction`.

Background and pointing are treated as `no_interaction`.

## Current Status

Completed:

- ChildPlay-Hand annotations and pose files explored
- Temporal sequence dataset generated
- Binary object-interaction model trained
- PyTorch model converted successfully to Core ML
- FastAPI inference server created
- Browser camera access tested
- MediaPipe 21-point hand tracking tested
- Live browser page connected to the model
- HTTPS access from an iPhone supported through Cloudflare Tunnel

Current deployment threshold:

```text
0.66
```

Current test performance at that threshold:

| Metric | Score |
|---|---:|
| Accuracy | 0.719 |
| Precision | 0.642 |
| Recall | 0.441 |
| F1 | 0.523 |

These results are sufficient for an MVP/prototyping workflow but are **not yet research- or deployment-grade**.

## Project Structure

Key files and directories:

```text
childplay_hand/
│
├── data/
│   ├── childplay_hand/
│   │   └── data/
│   │       ├── annotations/
│   │       │   ├── train/
│   │       │   ├── val/
│   │       │   └── test/
│   │       │
│   │       └── pose/
│   │           ├── cp_pose_train.pkl
│   │           ├── cp_pose_val.pkl
│   │           └── cp_pose_test.pkl
│   │
│   └── ChildPlay-gaze/
│       ├── videos.csv
│       ├── clips.csv
│       ├── annotations/
│       ├── videos/
│       ├── clips/
│       └── images/
│
├── mvp/
│   ├── build_temporal_dataset.py
│   ├── build_temporal_dataset_v2.py
│   ├── model_binary.py
│   ├── train_binary_mvp.py
│   ├── tune_binary_threshold.py
│   ├── convert_to_coreml.py
│   │
│   ├── dataset_v2/
│   │   ├── X_train.npy
│   │   ├── y_train.npy
│   │   ├── X_val.npy
│   │   ├── y_val.npy
│   │   ├── X_test.npy
│   │   ├── y_test.npy
│   │   ├── metadata_train.csv
│   │   ├── metadata_val.csv
│   │   └── metadata_test.csv
│   │
│   ├── models_binary/
│   │   ├── child_object_interaction.pt
│   │   ├── confusion_matrix.png
│   │   ├── confusion_matrix.csv
│   │   ├── training_history.csv
│   │   ├── normalization.npz
│   │   ├── threshold.txt
│   │   └── iphone_threshold.txt
│   │
│   ├── coreml/
│   │   ├── ChildObjectInteraction.mlpackage
│   │   └── model_config.json
│   │
│   └── webapp/
│       ├── server.py
│       ├── test_server.py
│       └── static/
│           ├── index.html
│           ├── camera-test.html
│           ├── mediapipe-test.html
│           └── live-model.html
│
├── grasp_annotations.csv
├── grasp_events.csv
├── good_grasp_events.csv
├── young_child_grasp_candidates.csv
├── verified_child_grasp_events.csv
├── infant_pilot_20.csv
├── pilot_grasp_annotations.csv
└── pilot_reannotation.csv
```

## Conda Environments

Three environments were used during development.

### `childplay_hand`

Used for dataset processing and model training.

```bash
conda activate childplay_hand
```

### `childplay_coreml`

Used for Core ML conversion and the FastAPI web server.

```bash
conda activate childplay_coreml
```

### `childplay_download`

Used for downloading source videos with `yt-dlp`.

```bash
conda activate childplay_download
```

## Dataset Representation

The current V2 dataset is stored in:

```text
mvp/dataset_v2/
```

Each sample has shape:

```text
20 × 69
```

where:

```text
20 = temporal frames
```

and the 69 features are:

```text
21 hand landmarks × (x, y, confidence) = 63
+
wrist x, wrist y                         = 2
+
bounding-box centre x, centre y          = 2
+
bounding-box width, height               = 2
------------------------------------------------
Total                                    = 69
```

The V2 dataset contained approximately:

```text
TRAIN: 26,526 sequences
VAL:    2,972 sequences
TEST:   3,129 sequences
```

## Rebuild the Temporal Dataset

Activate the ChildPlay environment:

```bash
conda activate childplay_hand
```

Then run:

```bash
python mvp/build_temporal_dataset_v2.py
```

The generated arrays and metadata are written to:

```text
mvp/dataset_v2/
```

## Train the Binary Model

Activate:

```bash
conda activate childplay_hand
```

Train:

```bash
python mvp/train_binary_mvp.py
```

The trained model is saved as:

```text
mvp/models_binary/child_object_interaction.pt
```

## Tune the Decision Threshold

Run:

```bash
python mvp/tune_binary_threshold.py
```

The selected web/iPhone MVP threshold is stored in:

```text
mvp/models_binary/iphone_threshold.txt
```

The current value is:

```text
0.66
```

## Core ML Conversion

A Core ML version was generated for possible future native iOS use.

Activate:

```bash
conda activate childplay_coreml
```

Run:

```bash
python mvp/convert_to_coreml.py
```

Output:

```text
mvp/coreml/ChildObjectInteraction.mlpackage
```

The current browser-based MVP does **not** require Core ML because inference runs on the Mac using the PyTorch model.

# Running the Web MVP

The simplest way to restart the system is to use **two Terminal windows**.

## Terminal 1 — Start FastAPI

From the repository root:

```bash
conda activate childplay_coreml

python -m uvicorn mvp.webapp.server:app \
  --host 0.0.0.0 \
  --port 8000
```

Expected startup output:

```text
Loading model...
Model loaded.
Threshold: 0.66
Serving frontend from:
.../mvp/webapp/static

Uvicorn running on http://0.0.0.0:8000
```

Keep this Terminal running.

## Local Health Check

Open:

```text
http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "model": "child_object_interaction",
  "sequence_length": 20,
  "feature_dimension": 69,
  "threshold": 0.66
}
```

Model configuration is available at:

```text
http://127.0.0.1:8000/config
```

## Browser Test Pages

### Camera only

```text
http://127.0.0.1:8000/app/camera-test.html
```

Use this to verify browser camera permissions.

### MediaPipe only

```text
http://127.0.0.1:8000/app/mediapipe-test.html
```

Use this to verify that the 21 hand landmarks are detected and drawn correctly.

### Full Live Model

```text
http://127.0.0.1:8000/app/live-model.html
```

This page performs:

```text
camera
→ MediaPipe
→ feature extraction
→ 20-frame buffer
→ /predict
→ PyTorch model
→ live probability
```

# Using an iPhone Camera

Mobile Safari needs HTTPS for camera access. A temporary Cloudflare Tunnel is used for the MVP.

## Terminal 2 — Start Cloudflare Tunnel

Install once:

```bash
brew install cloudflared
```

Start a temporary tunnel:

```bash
cloudflared tunnel --url http://localhost:8000
```

Cloudflare returns a temporary address similar to:

```text
https://random-words.trycloudflare.com
```

Keep this Terminal running.

On the iPhone, first verify the backend:

```text
https://YOUR-TUNNEL.trycloudflare.com/health
```

Then open:

```text
https://YOUR-TUNNEL.trycloudflare.com/app/live-model.html
```

Tap:

```text
Start Monitoring
```

and allow Safari camera access.

> The `trycloudflare.com` address is temporary. A new URL is normally generated each time the tunnel is restarted.

## Live Prediction Logic

When a hand is visible:

```text
MediaPipe
    ↓
21 landmarks
    ↓
69-feature vector
    ↓
rolling 20-frame buffer
    ↓
POST /predict
```

Predictions are requested roughly four times per second.

The model returns:

```json
{
  "interaction_probability": 0.73,
  "threshold": 0.66,
  "interaction": true,
  "label": "object_interaction"
}
```

The frontend does not immediately display a positive label from a single result.

It requires approximately:

```text
3 positive predictions out of the last 4
```

before showing:

```text
OBJECT INTERACTION
```

This temporal confirmation helps reduce unstable false detections.

## Quick Restart Procedure

After restarting the computer, the MVP can be launched as follows.

### Terminal 1

```bash
cd /path/to/childplay_hand
conda activate childplay_coreml

python -m uvicorn mvp.webapp.server:app \
  --host 0.0.0.0 \
  --port 8000
```

### Terminal 2

```bash
cloudflared tunnel --url http://localhost:8000
```

Copy the generated HTTPS URL and open on the phone:

```text
https://YOUR-TUNNEL.trycloudflare.com/app/live-model.html
```

Tap **Start Monitoring**.

## API Endpoints

### `GET /`

Basic service status.

### `GET /health`

Returns model and server status.

### `GET /config`

Returns configuration used by the frontend, including:

- sequence length
- feature dimension
- decision threshold
- confirmation settings
- confidence-channel fill values

### `POST /predict`

Input:

```json
{
  "sequence": [
    [69 values],
    [69 values]
  ]
}
```

Exactly 20 frames are required:

```text
[20, 69]
```

Output:

```json
{
  "interaction_probability": 0.73,
  "threshold": 0.66,
  "interaction": true,
  "label": "object_interaction"
}
```

## Privacy Notes

For the current design:

- camera video is processed in the browser by MediaPipe
- raw video is not intentionally sent to the PyTorch inference endpoint
- the browser sends the derived 20 × 69 feature sequence to the FastAPI server
- when Cloudflare Tunnel is used, traffic travels through the temporary HTTPS tunnel
- MediaPipe JavaScript/WASM/model resources are currently loaded from external CDN/Google-hosted resources

For a formal user study, privacy, data retention, consent, security, and hosting should be reviewed separately.

## Dataset and Redistribution

The repository should **not automatically redistribute ChildPlay raw videos or other restricted/source dataset material** unless the corresponding dataset licenses and usage terms explicitly permit redistribution.

Users should obtain required datasets from the original sources and follow their terms of use.

Large generated datasets, downloaded source videos, model checkpoints, and temporary outputs should normally be excluded from Git with `.gitignore` where appropriate.

## Known Limitations

The current model is an MVP.

Important limitations include:

- moderate precision and low recall at the selected threshold
- ChildPlay pose features and live MediaPipe features come from different pose-estimation systems
- object identity is not yet explicitly detected
- the classifier currently detects broad object interaction rather than fine-grained grasp type
- pincer vs palmar grasp classification was not sufficiently reliable in the initial naturalistic annotation pilot
- live phone-camera performance still requires systematic calibration and evaluation
- the system is not clinically validated

## Suggested Live Evaluation

Test the live system under several conditions:

```text
1. stationary open hand
2. moving empty hand
3. reaching toward an object
4. picking up an object
5. holding an object
6. releasing an object
7. different object sizes
8. different camera distances
9. different participants
10. different lighting/backgrounds
```

Record the interaction probability for each condition.

This can be used to quantify the domain shift between:

```text
ChildPlay training poses
        ↓
MediaPipe live poses
```

and guide recalibration or fine-tuning.

## Planned Next Steps

The intended next stages are:

```text
Live interaction detection
        ↓
calibration with real phone-camera data
        ↓
improve interaction classifier
        ↓
detect / understand manipulated object
        ↓
combine movement + object + context
        ↓
generate caregiver play suggestion
        ↓
optional spoken feedback
```

Longer term, the system can be evaluated as a **context-aware caregiver-support tool for naturalistic child play**, rather than as a diagnostic system.

## Research Direction

A possible framing for the project is:

> **Can a lightweight AI system recognize children's naturalistic hand-object interactions in real time and use these observations to provide context-sensitive play support to caregivers?**

Potential evaluation dimensions include:

- recognition accuracy
- precision / recall / F1
- mobile/web latency
- false positive rate
- robustness across users and objects
- caregiver usefulness
- trust
- perceived intrusiveness
- appropriateness of generated suggestions

## Disclaimer

This project is an experimental research prototype. It is not intended to diagnose developmental delay, assess a child's clinical condition, or replace professional developmental evaluation.
