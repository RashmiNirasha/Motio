# Setup Instructions

This guide sets up the environment for the ChildPlay-Hand project.

---

## 1. Create and Activate Conda Environment

```bash
conda create -n childplay_hand python=3.8 -y
conda activate childplay_hand
```

## 2. Install PyTorch with CUDA

```shell
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

## 3. Install OpenMIM and Dependencies

```shell
pip install -U openmim==0.3.9
conda install fsspec -c conda-forge
mim install "mmengine==0.10.5"
mim install "mmcv==2.1.0"
mim install "mmdet==3.3.0"
```

## 4. Install MMPose (from source)

```shell
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
cd ..
```

## 5. Install Project Requirements

```shell
pip install -r requirements.txt
```

## 6. Install PySKL (for skeleton-based models)

```shell
git clone https://github.com/kennymckormick/pyskl.git
cd pyskl
pip install -e .
cd ..
```