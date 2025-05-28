#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import torch
import torch.optim as optim
from omegaconf import DictConfig

from src.models.base_model_tas import Base_Lit_Model
from src.models.components.mstcn import MSTCN, MSTCN_LOSS
from src.utils import get_pylogger

log = get_pylogger(__name__)


class MSTCN_Model(Base_Lit_Model):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # Model
        self.model = MSTCN(self.cfg)
        # Loss Function
        self.loss_func = MSTCN_LOSS(self.cfg)

    def configure_optimizers(self):
        # Following original MS-TCN implementation: https://github.com/yabufarha/ms-tcn
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.solver.lr)
        return optimizer

    def squeeze_batch(self, x):
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.squeeze(0)
            else:
                x[k] = v

    def forward(self, x, return_last_stage):
        self.squeeze_batch(x)
        return self.model(x, return_last_stage)
