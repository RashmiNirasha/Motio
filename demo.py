#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import os
from typing import Tuple

import hydra
import lightning as L
import pyrootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

from src import utils
from src.datamodules.video_datamodule import VideoDataModule
from src.utils.pose_extraction import run_pose_extraction_demo
from src.utils.tracking import track_people
from src.utils.vis_tools import draw_on_video

# Set precision for matrix multiplication
torch.set_float32_matmul_precision("medium")

# Logger setup
log = utils.get_pylogger(__name__)

# Root setup
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


def demo(cfg: DictConfig) -> Tuple[dict, dict]:
    """Run the full demo pipeline including extraction, inference, and visualization."""
    assert not cfg.train
    video_path = cfg.video_path
    output_path = cfg.output_path

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    assert os.path.exists(video_path)
    os.makedirs(output_path, exist_ok=True)

    # Tracking
    path_to_tracks = track_people(video_path, output_path)

    # Pose Extraction
    path_to_pose = run_pose_extraction_demo(video_path, path_to_tracks, output_path)

    # Hand Action Recogintion
    datamodule: LightningDataModule = VideoDataModule(cfg.configs, False, video_path, path_to_pose)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=None, logger=None)

    ckpt_path = cfg.get("ckpt_path")
    assert ckpt_path is not None and os.path.exists(ckpt_path), "Checkpoint path is invalid!"

    trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    path_to_action = f"{output_path}/pred.pkl"

    # Draw predictions
    draw_on_video(cfg.configs, video_path, path_to_pose, path_to_action, output_path)


@hydra.main(config_path="./configs", config_name="demo.yaml", version_base="1.2")
def main(cfg: DictConfig):
    demo(cfg)


if __name__ == "__main__":
    main()
