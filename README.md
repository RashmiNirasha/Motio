# Motio: Child Hand-Object Interaction and Pincer Grasp Monitoring

A research prototype for **real-time monitoring of children's hand-object interaction**, built on the ChildPlay-Hand dataset, MediaPipe Hand Landmarker, lightweight temporal PyTorch models, FastAPI, and a browser camera interface.

> **Research prototype only.** This is not a medical device. It must not be used to diagnose motor-development conditions or make clinical decisions.

## What it does

The live page tracks **up to two hands** and, for each hand, shows:

- **Object interaction**: whether the hand is likely manipulating an object (trained on ChildPlay-Hand).
- **Pincer grasp**: whether the hand is holding something between thumb and index finger. This uses a rule-based hand-shape check until a trained pincer model exists (see [Pincer grasp workflow](#pincer-grasp-workflow)).

```text
Phone / laptop camera
        ↓
MediaPipe Hand Landmarker (2 hands, 21 landmarks each)
        ↓
per-hand 20-frame buffer
        ↓
POST /predict_hand  (FastAPI)
        ├── interaction model   (69 features per frame)
        └── pincer model / rule (74 features per frame)
        ↓
per-hand OBJECT INTERACTION / PINCER GRASP
```

A label is only shown after 3 of the last 4 predictions agree, to reduce flicker.

## Quick start

Environment and data setup are in [SETUP.md](SETUP.md). Once set up:

```bash
conda activate childplay_coreml
python -m uvicorn mvp.webapp.server:app --host 0.0.0.0 --port 8000
```

Open **http://127.0.0.1:8000/app/**, which links to every page:

| Page | Purpose |
|---|---|
| `live-model.html` | Live two-hand monitoring (interaction + pincer) |
| `label-pincer.html` | Label ChildPlay hand clips for pincer training |
| `record-pincer.html` | Record your own labelled clips from a camera or video file |
| `camera-test.html` | Check camera permissions |
| `mediapipe-test.html` | Check MediaPipe hand landmarks |

### Using an iPhone camera

Mobile Safari needs HTTPS for camera access. In a second terminal:

```bash
cloudflared tunnel --url http://localhost:8000
```

Open `https://<your-tunnel>.trycloudflare.com/app/live-model.html` on the phone and tap **Start Monitoring**. Choose **Back camera** to film a child. The tunnel URL changes every time it restarts.

## Object-interaction model

ChildPlay actions `grasp`, `hold`, `release`, `operate`, `put`, `give`, `drop` and `throw` are grouped as `object_interaction`; background and pointing are `no_interaction`.

Each sample is 20 frames × 69 features: 21 landmarks × (x, y, confidence), wrist x/y, bounding-box centre x/y and width/height.

Test performance at the deployed threshold (0.66):

| Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.719 | 0.642 | 0.441 | 0.523 |

Good enough for prototyping, but not research- or deployment-grade. Rebuild / retrain steps are in [SETUP.md](SETUP.md).

## Pincer grasp workflow

ChildPlay-Hand labels actions, not grasp types, so pincer labels have to come from a person. The pipeline uses ChildPlay to find **candidate moments** and you label them.

1. **Extract candidates**: grasp/hold/operate moments where the hand is large (≥ 90 px) and its keypoints are confident. **Children only**, using ChildPlay-Gaze's `is_child` annotation.

   ```bash
   conda activate childplay_hand
   python mvp/extract_pincer_candidates.py
   ```

   More candidates are in source videos that aren't downloaded yet:

   ```bash
   conda activate childplay_download
   bash mvp/download_childplay_videos.sh
   ```

   Then re-run the extractor. Existing labels are kept.

2. **Label** at `/app/label-pincer.html`:
   - `1` pincer · `2` other grasp · `3` no grasp · `4` unclear (not used for training)
   - `Y` / `O` / `N` set the video's age band: under 5, 5 or older, or not sure. ChildPlay has no ages; each video shows a hint from its title.

   MediaPipe runs on each clip in the browser, so training uses the same landmarks as the live app.

3. **Evaluate**, without overwriting the saved model:

   ```bash
   conda activate childplay_hand
   python mvp/train_pincer.py --under-5-only --cross-validate --no-save
   ```

   Leave-one-video-out cross-validation tests on every video in turn, and compares the model with the rule-based check on the same clips.

4. **Train and deploy** once the model clearly beats the rule:

   ```bash
   python mvp/train_pincer.py --under-5-only --cross-validate
   ```

   Then restart the server. The live page switches from the rule to the model automatically.

Clips recorded with `/app/record-pincer.html` are used in training too. Use one session name per child or video, because training keeps each session entirely in one split.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health`, `GET /config` | Server status; thresholds and settings used by the frontend |
| `POST /predict` | Interaction only. Body: `{"sequence": [20 × 69]}` |
| `POST /predict_hand` | Interaction + pincer for one hand. Body: `{"sequence": [20 × 69], "landmarks": [20 × 21 × 3]}` |
| `POST /record_pincer`, `GET /pincer_data_stats` | Save / count recorder clips |
| `GET /pincer_candidates`, `POST /pincer_label`, `POST /video_age` | Labelling page |

Example `/predict_hand` response:

```json
{
  "interaction_probability": 0.73,
  "interaction": true,
  "threshold": 0.66,
  "pincer_probability": 0.12,
  "pincer": false,
  "pincer_threshold": 0.5,
  "pincer_source": "heuristic"
}
```

## Repository layout

```text
mvp/
├── build_temporal_dataset_v2.py   ChildPlay → interaction dataset (dataset_v2/)
├── model_binary.py                interaction model
├── train_binary_mvp.py            train interaction model (models_binary/)
├── tune_binary_threshold.py       pick deployment threshold
├── convert_to_coreml.py           optional Core ML export (coreml/)
├── pincer_features.py             pincer features + rule-based check
├── model_pincer.py                pincer model
├── train_pincer.py                train / evaluate pincer model (models_pincer/)
├── extract_pincer_candidates.py   ChildPlay → child hand clips for labelling
├── age_hints.py                   age hint from video titles
├── download_childplay_videos.sh   fetch missing ChildPlay source videos
└── webapp/
    ├── server.py                  FastAPI backend + static pages
    ├── test_server.py             API smoke test
    └── static/                    live, labelling, recorder and test pages
src/, configs/, train.py, test.py  original ChildPlay-Hand baseline code (authors')
data/                              ChildPlay data (not in Git, see data/README.md)
```

Not in Git (generated or private): ChildPlay data, extracted crops (`mvp/pincer_candidates/`), labelled clips (`mvp/pincer_data/`), and model checkpoints (`*.pt`).

## Privacy

- Camera video is processed in the browser by MediaPipe; only derived hand landmarks/features are sent to the server.
- Through Cloudflare Tunnel, that traffic passes through a temporary HTTPS tunnel.
- MediaPipe's JavaScript, WASM and model files load from external CDNs (jsDelivr, Google).
- Labelled clips contain children's hand landmarks and are kept out of Git.

Before a formal user study, review privacy, consent, data retention, security and hosting separately.

## Known limitations

- The interaction model has moderate precision and low recall at the chosen threshold.
- The interaction model is trained on ChildPlay's pose estimator but runs on MediaPipe landmarks live (domain shift).
- Pincer detection currently uses a rule-based check. The first trained pincer model (7 pincer clips) did not beat it, and more labelled clips are needed.
- ChildPlay hands are often small, so fine finger detail is limited, and MediaPipe misses some of them.
- Children's ages are not annotated; under-5 filtering relies on per-video confirmation.
- Object identity is not modelled.
- Not clinically validated.

## Research direction

> Can a lightweight AI system recognise children's naturalistic hand-object interactions and fine motor skills, such as the pincer grasp, in real time, and use these observations to give caregivers context-sensitive play support?

Planned steps: calibrate with real phone-camera data, then improve the classifiers, then understand the manipulated object, then generate caregiver play suggestions. The intended use is a **caregiver-support tool for naturalistic play**, not a diagnostic system.

## Dataset and attribution

This project builds on the **ChildPlay-Hand** dataset by Arya Farkhondeh, Samy Tafasca and Jean-Marc Odobez:

- *ChildPlay-Hand: A Dataset of Hand Manipulations in the Wild*, ECCV Workshops 2024. https://arxiv.org/abs/2409.09319
- Related: *ChildPlay: A New Benchmark for Understanding Children's Gaze Behaviour*, ICCV 2023. https://openaccess.thecvf.com/content/ICCV2023/html/Tafasca_ChildPlay_A_New_Benchmark_for_Understanding_Children%27s_Gaze_Behaviour_ICCV_2023_paper.html

The data remains the property of its authors. This repository does not redistribute ChildPlay videos or annotations; obtain them from the original sources and follow their terms. The baseline code in `src/` and `configs/` is the authors' (see `LICENSES/`). The MVP and analysis are my own work built on their dataset.
