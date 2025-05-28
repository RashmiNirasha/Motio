#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import math

import torch
import torch.optim as optim
from omegaconf import DictConfig

from src.models.base_model import Base_Lit_Model
from src.models.components.action_head import HandHead
from src.models.components.hiera import Hiera
from src.utils import get_pylogger

log = get_pylogger(__name__)


class Hiera_Model(Base_Lit_Model):
    """
    https://github.com/dbolya/hiera/blob/v0.2.0/hiera/train/engine.py
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # Encoder
        self.encoder = Hiera(cfg)

        # Decoder
        self.decoder = HandHead(cfg)

    def configure_optimizers(self):
        # optimizer
        self.lr_scaler = (
            self.trainer.num_devices
            * self.trainer.num_nodes
            * self.trainer.accumulate_grad_batches
            * self.cfg.train_batch_size
            / self.cfg.solver.lr_batch_size
        )
        optim_params = [
            {
                "params": filter(lambda p: p.requires_grad, self.encoder.parameters()),
                "lr": self.cfg.solver.lr_backbone * self.lr_scaler,
            },
            {
                "params": filter(lambda p: p.requires_grad, self.decoder.parameters()),
                "lr": self.cfg.solver.lr_head * self.lr_scaler,
            },
        ]
        optimizer = optim.AdamW(
            params=optim_params, weight_decay=self.cfg.solver.weight_decay, betas=(0.9, 0.95)
        )

        # scheduler
        def warm_start_and_cosine_annealing(epoch):
            if epoch < self.cfg.solver.warmup_epochs:
                lr = (epoch + 1) / self.cfg.solver.warmup_epochs
            else:
                lr = 0.5 * (
                    1.0
                    + math.cos(
                        math.pi
                        * ((epoch + 1) - self.cfg.solver.warmup_epochs)
                        / (self.trainer.max_epochs - self.cfg.solver.warmup_epochs)
                    )
                )
            return lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[warm_start_and_cosine_annealing for _ in range(len(optim_params))],
            verbose=False,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def forward_extraction(self, x):
        feat = self.encoder(x)
        return {"feat": feat, **self.decoder(feat)}

    def forward(self, x):
        return self.decoder(self.encoder(x))
