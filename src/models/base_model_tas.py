#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

from typing import Any

import torch
from lightning import LightningModule
from omegaconf import DictConfig
from torchmetrics import MaxMetric, MeanMetric

from src.evaluation.cp_hand_eval import CP_evaluator
from src.models.components.losses import HandLoss
from src.utils import get_pylogger

log = get_pylogger(__name__)


class Base_Lit_Model(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cfg = self.hparams.cfg
        self.tasks = cfg.task.split("_")

        self.model = None

        # For averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # Metrics
        self.val_evaluator = CP_evaluator(cfg, "val")
        self.test_evaluator = CP_evaluator(cfg, "test")
        self.val_best = MaxMetric()
        self.monitor_metric = cfg.monitor_metric
        self.monitor_task = cfg.monitor_task
        assert self.monitor_task in self.tasks

        # Loss func
        self.loss_func = HandLoss(cfg)

    def configure_optimizers(self):
        raise

    def forward(self, x):
        raise

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------
    def on_train_start(self):
        torch.cuda.empty_cache()
        self.val_loss.reset()
        self.val_best.reset()
        self.val_evaluator.reset()

    def step(self, batch: Any):
        output = self.forward(batch, return_last_stage=False)
        loss_dict = self.loss_func(output, batch)

        return loss_dict

    def training_step(self, input_batch: Any, batch_idx: int):
        batch, _ = input_batch
        loss_dict = self.step(batch)
        loss = sum([v for k, v in loss_dict.items()])
        self.train_loss(loss.item())
        # logging
        for key in loss_dict.keys():
            self.log(
                "train/loss/" + key,
                loss_dict[key].item(),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def on_train_epoch_end(self):
        log.info("Training epoch " + str(self.current_epoch) + " ended.")
        self.trainer.datamodule.data_train.reset()

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------
    def on_validation_start(self):
        torch.cuda.empty_cache()

    def post_process_output_on_inference(self, x):
        return x

    def validation_step(self, input_batch, batch_idx: int):
        batch, batch_meta = input_batch
        output, output_ls = self.forward(batch, return_last_stage=True)
        loss_dict = self.loss_func(output, batch)
        loss = sum([v for k, v in loss_dict.items()])
        self.val_loss(loss.item())
        # logging
        for key in loss_dict.keys():
            self.log(
                "val/loss/" + key,
                loss_dict[key].item(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        output = self.post_process_output_on_inference(output)
        # store sequence results
        self.val_evaluator.store_results_sequence(batch, batch_meta, output_ls)

    def on_validation_epoch_end(self) -> None:
        # compute metrics
        res_metrics = self.val_evaluator.compute_metrics()
        monitor_metric = res_metrics[self.monitor_task]["segment"][self.monitor_metric]

        # Monitor best
        self.val_best(monitor_metric)
        self.log("val/monitor_metric", monitor_metric, prog_bar=True)
        self.log("val/monitor_metric_best", self.val_best.compute(), prog_bar=True)

        self._log_metrics(res_metrics, "val")
        self.trainer.datamodule.data_val.reset()

    # -------------------------------------------------------------
    # Test
    # -------------------------------------------------------------
    def on_test_start(self):
        torch.cuda.empty_cache()

    def test_step(self, input_batch, batch_idx: int):
        batch, batch_meta = input_batch
        output, output_ls = self.forward(batch, return_last_stage=True)
        output = self.post_process_output_on_inference(output)
        # store batch sequence
        self.test_evaluator.store_results_sequence(batch, batch_meta, output_ls)

    def on_test_epoch_end(self) -> None:
        # compute metrics
        res_metrics = self.test_evaluator.compute_metrics()
        self._log_metrics(res_metrics, "test")
        self.trainer.datamodule.data_test.reset()

    def _log_metrics(self, res_metrics, stage):
        # log metrics
        for task in self.tasks:
            for type_ in ["frame", "segment"]:
                for k, v in res_metrics[task][type_].items():
                    if k.startswith(("classification_report", "confusion", "precision", "recall")):
                        continue
                    if isinstance(v, (int, float)):
                        self.log(f"{stage}/{task}/{type_}/{k}", v, prog_bar=True)
