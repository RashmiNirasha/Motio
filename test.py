#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

from typing import Optional, Tuple

import hydra
import lightning as L
import pyrootutils
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

from src import utils

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


def test(cfg: DictConfig) -> Tuple[dict, dict]:
    """Run model testing using the provided configuration.

    Args:
        cfg (DictConfig): Configuration from Hydra.
    """
    log.info(f"Experiment folder: {cfg.paths.exp_dir}")
    print(f"Experiment folder: {cfg.paths.exp_dir}")

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Instantiate DataModule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # Instantiate Model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Testing
    log.info("Starting testing!")
    ckpt_path = cfg.get("ckpt_path")
    log.info(f"Best checkpoint path: {ckpt_path}")

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=None)
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


@hydra.main(version_base="1.2", config_path="./configs", config_name="test.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main function to execute testing."""
    test(cfg)


if __name__ == "__main__":
    main()
