# Setup and Run Guide

This project is a research MVP for real-time child hand-object interaction monitoring using the ChildPlay-Hand dataset, MediaPipe, a temporal PyTorch model, FastAPI, and a browser-based camera interface.

> This is a prototyping and analysis project, not a medical device.

---

## 1. Clone the repository

```bash
git clone https://github.com/RashmiNirasha/Motio.git
cd childplay_hand
```

---

## 2. Create the main dataset/training environment

Use the primary environment for dataset exploration, temporal dataset generation, and model training.

```bash
conda create -n childplay_hand python=3.10 -y
conda activate childplay_hand
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

If you are reproducing the original ChildPlay model pipeline, ensure the environment includes PyTorch and the required scientific Python stack from the repository requirements.

---

## 3. Create the Core ML / FastAPI environment

This environment is used for the browser-based MVP, the FastAPI server, and the Core ML conversion step.

```bash
conda create -n childplay_coreml python=3.11 -y
conda activate childplay_coreml
```

Install the runtime dependencies needed for the server and model serving:

```bash
pip install fastapi uvicorn numpy pandas torch
```

If you want the full repository environment to match the development setup, install the project requirements as well:

```bash
pip install -r requirements.txt
```

---

## 4. Create the download environment

This environment is used for source-video downloading tasks when needed.

```bash
conda create -n childplay_download python=3.10 -y
conda activate childplay_download
```

Install download tooling used by the project:

```bash
pip install yt-dlp
```

---

## 5. Rebuild the temporal dataset

From the repository root:

```bash
conda activate childplay_hand
python mvp/build_temporal_dataset_v2.py
```

This writes the generated dataset arrays and metadata to:

```text
mvp/dataset_v2/
```

The current dataset format is:

```text
20 x 69
```

where the 69 features are:

- 21 hand landmarks × (x, y, confidence) = 63 features
- wrist x, wrist y = 2 features
- bounding-box center x, center y = 2 features
- bounding-box width, height = 2 features

---

## 6. Train the binary interaction model

```bash
conda activate childplay_hand
python mvp/train_binary_mvp.py
```

The trained model is saved at:

```text
mvp/models_binary/child_object_interaction.pt
```

---

## 7. Tune the decision threshold

```bash
conda activate childplay_hand
python mvp/tune_binary_threshold.py
```

The chosen web/iPhone MVP threshold is stored in:

```text
mvp/models_binary/iphone_threshold.txt
```

The current threshold is:

```text
0.66
```

---

## 8. Convert to Core ML (optional)

This is optional and mainly relevant to future native iOS deployment work.

```bash
conda activate childplay_coreml
python mvp/convert_to_coreml.py
```

The generated model package is written to:

```text
mvp/coreml/ChildObjectInteraction.mlpackage
```

---

## 9. Start the FastAPI app for the MVP

Use two terminals.

### Terminal 1 — start the API server

```bash
cd /path/to/childplay_hand
conda activate childplay_coreml

python -m uvicorn mvp.webapp.server:app \
  --host 0.0.0.0 \
  --port 8000
```

Expected startup output includes:

```text
Loading model...
Model loaded.
Threshold: 0.66
Serving frontend from:
.../mvp/webapp/static
```

Then verify the backend locally:

```text
http://127.0.0.1:8000/health
```

and:

```text
http://127.0.0.1:8000/config
```

---

## 10. Open the browser test pages

Once the server is running, open the pages below in the browser:

### Camera only

```text
http://127.0.0.1:8000/app/camera-test.html
```

### MediaPipe landmarks only

```text
http://127.0.0.1:8000/app/mediapipe-test.html
```

### Full live model

```text
http://127.0.0.1:8000/app/live-model.html
```

This page performs:

```text
camera -> MediaPipe -> 69-feature extraction -> 20-frame buffer -> /predict -> PyTorch model -> probability
```

---

## 11. Use the live model on an iPhone

Mobile Safari requires HTTPS for camera access. Use a temporary Cloudflare tunnel.

### Install Cloudflare Tunnel once

```bash
brew install cloudflared
```

### Start the tunnel

```bash
cloudflared tunnel --url http://localhost:8000
```

This returns a temporary HTTPS address, for example:

```text
https://random-words.trycloudflare.com
```

Open the following on the iPhone:

```text
https://YOUR-TUNNEL.trycloudflare.com/app/live-model.html
```

Then tap:

```text
Start Monitoring
```

and allow camera access.

---

## 12. Quick restart workflow

After restarting the computer:

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

Then use the generated HTTPS URL on the phone to open the live model page.

---

## 13. API contract

### `GET /`

Basic service status.

### `GET /health`

Returns model status and configuration metadata.

### `GET /config`

Returns:

- sequence length
- feature dimension
- threshold
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

Example output:

```json
{
  "interaction_probability": 0.73,
  "threshold": 0.66,
  "interaction": true,
  "label": "object_interaction"
}
```

---

## 14. Dataset and redistribution notes

This project depends on the ChildPlay-Hand dataset from the original researchers. The repository should not redistribute raw videos or source dataset materials unless the relevant data licensing and terms permit redistribution.

Users should download required data from the original sources and follow the dataset's usage terms.

Large generated datasets, downloaded source videos, model checkpoints, and temporary outputs should usually be excluded from Git where appropriate.

---

## 15. Known limitations

The current model is a prototype MVP and has the following limitations:

- moderate precision and lower recall at the chosen threshold
- different pose-estimation systems are used for training data vs live MediaPipe data
- object identity is not explicitly modeled
- the classifier detects broad object interaction rather than fine-grained grasp types
- live phone-camera performance still needs systematic evaluation
- the system is not clinically validated

---

## 16. Recommended evaluation checklist

Test the model under several conditions:

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

Record the interaction probability and confirm whether the decision remains stable across these scenarios.