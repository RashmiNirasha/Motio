#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import os
from typing import List, Optional, Tuple

import hydra
import lightning as L
import pyrootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
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


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Train the model and optionally evaluate on a test set.

    Wrapped with @task_wrapper to apply additional utilities before and after execution.

    Args:
        cfg (DictConfig): Configuration from Hydra.

    Returns:
        Tuple[dict, dict]: Metrics dictionary and instantiated objects dictionary.
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

    # Instantiate Callbacks
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    # Instantiate Logger
    logger = None
    if cfg.logger.log:
        log.info("Instantiating loggers...")
        logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    # Instantiate Trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # Store all objects
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    # Log hyperparameters if a logger is used
    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    # Determine checkpoint path
    checkpoint_path = cfg.get("ckpt_path")
    if checkpoint_path is None:
        last_ckpt = os.path.join(cfg.configs.storage_folder, "checkpoints/last.ckpt")
        checkpoint_path = last_ckpt if os.path.exists(last_ckpt) else None
        log.info(
            f"Loading weights from last checkpoint: {checkpoint_path}"
        ) if checkpoint_path else None
    elif not os.path.exists(checkpoint_path):
        checkpoint_path = None
    else:
        log.info(f"Loading weights from checkpoint: {checkpoint_path}")

    # Training
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)

    train_metrics = trainer.callback_metrics

    # Testing
    test_metrics = {}
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = (
            trainer.checkpoint_callback.best_model_path
            if cfg.get("train")
            else cfg.get("ckpt_path")
        )

        if not ckpt_path:
            log.warning("Best checkpoint not found! Using current weights for testing...")
            ckpt_path = None

        trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

        log.info(f"Best checkpoint path: {ckpt_path}")
        test_metrics = trainer.callback_metrics

    return {**train_metrics, **test_metrics}, object_dict


@hydra.main(version_base="1.2", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main function to execute training."""
    metric_dict, _ = train(cfg)
    return utils.get_metric_value(metric_dict, metric_name=cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
