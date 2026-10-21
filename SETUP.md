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

### `childplay_coreml`: web server and Core ML

```bash
conda create -n childplay_coreml python=3.11 -y
conda activate childplay_coreml
pip install fastapi uvicorn numpy torch
pip install coremltools   # only for mvp/convert_to_coreml.py
```

### `childplay_download`: source videos

```bash
conda create -n childplay_download python=3.11 -y
conda activate childplay_download
pip install yt-dlp
```

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

## 4. Interaction model (optional rebuild)

The prebuilt dataset (`mvp/dataset_v2/`) and normalization files are in Git. Only the checkpoint `child_object_interaction.pt` is not, so train it once:

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

http://127.0.0.1:8000/health should return `"status": "ok"`. With the server running, `python mvp/webapp/test_server.py` sends two test sequences to `/predict`.
