# Setup Guide

Environment and data setup for Motio. For how to run and use the system, see [README.md](README.md).

## 1. Clone

```bash
git clone https://github.com/RashmiNirasha/Motio.git
cd Motio
git lfs pull   # dataset arrays, Core ML weights and figures are stored with Git LFS
```

## 2. Conda environments

Three environments are used.

### `childplay_hand`: datasets and training

```bash
conda create -n childplay_hand python=3.8 -y
conda activate childplay_hand
pip install -r requirements.txt
pip install torch
```

`requirements.txt` also covers the original ChildPlay-Hand baseline code in `src/`.

### `childplay_coreml`: web server, object detector, AI assistant

```bash
conda create -n childplay_coreml python=3.11 -y
conda activate childplay_coreml
pip install -r mvp/requirements-mvp.txt
yolo settings sync=False hub=False   # turn off Ultralytics usage analytics
```

`mvp/requirements-mvp.txt` pins the versions v1 was tested with. PyTorch and NumPy are
pinned on purpose: an unpinned detector install can upgrade them and break the hand model.
On first start the server downloads the YOLO-World weights (~26 MB, into
`mvp/models_detect/`) and the CLIP ViT-B/32 text encoder (~350 MB, into `~/.cache/clip`).

### `childplay_download`: source videos

```bash
conda create -n childplay_download python=3.11 -y
conda activate childplay_download
pip install yt-dlp
```

### AI assistant: Claude API or a free local model

Either one is enough.

- **Claude API**: create a key at console.anthropic.com and set it in the terminal that
  starts the server: `export ANTHROPIC_API_KEY="..."`. Never commit it.
- **Local and free**: install [Ollama](https://ollama.com), then `ollama pull qwen2.5:7b`
  (~4.7 GB) and start the server with `MOTIO_LLM_BACKEND=ollama`. Ollama 0.5 or newer is
  recommended, because it enforces the reply schema.

### Cloudflare Tunnel (for the iPhone camera)

```bash
brew install cloudflared
```

## 3. ChildPlay data

The data is not in Git. Download it as described in [data/README.md](data/README.md), so that you have:

```text
data/
├── childplay_hand/data/
│   ├── annotations/{train,val,test}/*.csv
│   └── pose/cp_pose_{train,val,test}.pkl
└── ChildPlay-gaze/
    ├── annotations/{train,val,test}/*.csv   (provides is_child)
    ├── clips.csv
    ├── videos.csv
    └── videos/*.mp4
```

If the data already lives elsewhere, symlink it instead of copying:

```bash
ln -s /path/to/childplay_hand data/childplay_hand
ln -s /path/to/ChildPlay-gaze data/ChildPlay-gaze
```

Only `annotations` and `pose` are needed to rebuild the interaction model. The source `videos/` are needed only for pincer candidate extraction (`mvp/download_childplay_videos.sh` fetches missing ones).

## 4. Interaction models

Model checkpoints (`*.pt`) are not in Git. The prebuilt dataset (`mvp/dataset_v2/`) is,
so both models can be trained without downloading ChildPlay.

**Live model (used by v1):** hand shape only, so it works on MediaPipe landmarks.
About 17 minutes on an Apple M3 CPU:

```bash
conda activate childplay_hand
python mvp/train_live_interaction.py   # → mvp/models_live/child_interaction_live.pt
```

**Original model** (kept for comparison; start the server with
`MOTIO_INTERACTION_MODEL=original` to use it):

```bash
conda activate childplay_hand
python mvp/train_binary_mvp.py        # → mvp/models_binary/child_object_interaction.pt
python mvp/tune_binary_threshold.py   # → mvp/models_binary/iphone_threshold.txt
```

To rebuild the dataset from ChildPlay first:

```bash
python mvp/build_temporal_dataset_v2.py   # → mvp/dataset_v2/
```

## 5. Core ML export (optional)

For future native iOS work only; the web app does not need it.

```bash
conda activate childplay_coreml
python mvp/convert_to_coreml.py   # → mvp/coreml/ChildObjectInteraction.mlpackage
```

## 6. Pincer model

There is no pretrained pincer model. The server uses a rule-based check until you label data and train one. See [Pincer grasp workflow](README.md#pincer-grasp-workflow) in the README.

## 7. Check the server

```bash
conda activate childplay_coreml
python -m uvicorn mvp.webapp.server:app --host 0.0.0.0 --port 8000
```

http://127.0.0.1:8000/health should return `"status": "ok"` and
`"model": "child_interaction_live (hand shape only)"`. The log prints
`Object detector ready on mps` about 10 seconds after start.

Automated tests (no camera, no API key needed):

```bash
python mvp/observe/test_hand_state.py
python mvp/observe/test_assistant_pipeline.py
```
