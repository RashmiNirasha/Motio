#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import torch
import torch.nn as nn
import torch.optim as optim
from mmcv.runner import load_checkpoint
from omegaconf import DictConfig

from src.models.base_model import Base_Lit_Model
from src.models.components.action_head import HandHead
from src.models.components.rgbposeconv3d import Conv3DAdapter, RGBPoseConv3D
from src.utils import get_pylogger

log = get_pylogger(__name__)


class RGBPoseConv3D_Model(Base_Lit_Model):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        #  Input Adapter
        if cfg.input_type == "hand":
            self.adapter = Conv3DAdapter(
                in_channels=cfg.adapter.in_channels,
                out_channels=cfg.adapter.out_channels,
            )
        else:
            self.adapter = nn.Identity()
        # Encoder
        self.encoder = RGBPoseConv3D()
        # Decoders
        self.decoder_rgb = HandHead(cfg, feat_dim=2048, hidden_dim=cfg.head.hiddem_dim_rgb)
        self.decoder_pose = HandHead(cfg, feat_dim=512, hidden_dim=cfg.head.hiddem_dim_pose)

        # Initialize Weights
        self._init_weights()

    def _init_weights(self):
        # https://mmcv.readthedocs.io/en/master/_modules/mmcv/runner/checkpoint.html
        assert self.cfg.load_from
        revise_keys = [(r"^module.", "")]
        load_checkpoint(
            self.encoder,
            self.cfg.load_from,
            map_location="cpu",
            logger=log,
            revise_keys=revise_keys,
            strict=False,
        )

    def configure_optimizers(self):
        """
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        self.lr_scaler = (
            self.trainer.num_devices
            * self.trainer.num_nodes
            * self.trainer.accumulate_grad_batches
            * self.cfg.train_batch_size
            / self.cfg.solver.lr_batch_size
        )
        optim_params = [
            {
                "params": filter(lambda p: p.requires_grad, self.adapter.parameters()),
                "lr": self.cfg.solver.lr_adapter * self.lr_scaler,
            },
            {
                "params": filter(lambda p: p.requires_grad, self.encoder.parameters()),
                "lr": self.cfg.solver.lr_backbone * self.lr_scaler,
            },
            {
                "params": filter(lambda p: p.requires_grad, self.decoder_rgb.parameters()),
                "lr": self.cfg.solver.lr_head * self.lr_scaler,
            },
            {
                "params": filter(lambda p: p.requires_grad, self.decoder_pose.parameters()),
                "lr": self.cfg.solver.lr_head * self.lr_scaler,
            },
        ]

        # optimizer
        optimizer = optim.SGD(
            params=optim_params,
            momentum=self.cfg.solver.momentum,
            weight_decay=self.cfg.solver.weight_decay,
        )

        # scheduler
        def combined_lr_scheduler(epoch):
            warmup_epochs = self.cfg.solver.warmup_epochs
            base_lr = 1
            if epoch < warmup_epochs:
                lr = (epoch + 1) / warmup_epochs
            else:
                reduction_factor = self.cfg.solver.gamma
                lr = base_lr
                for milestone in self.cfg.solver.step_size:
                    if epoch + 1 >= milestone:
                        lr *= reduction_factor
                    else:
                        break
            return lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[combined_lr_scheduler for _ in range(len(optim_params))],
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

    def post_process_output_on_inference(self, res):
        """Late fusion."""
        hands = ["left_", "right_"] if self.cfg.input_type == "body" else [""]

        for task in self.tasks:
            for hand in hands:
                key = f"{hand}pred_{task}"
                if self.cfg.fusion_method == "rgb_pose":
                    res[key] = res["pose"][key] + res["rgb"][key]
                else:
                    res[key] = res[self.cfg.fusion_method][key]

        res.pop("pose", None)
        res.pop("rgb", None)

        return res

    def forward(self, x):
        res = dict()
        # encoding
        enc = self.encoder(self.adapter(x))
        # decoding RGB
        res["rgb"] = self.decoder_rgb(enc["e_rgb"])
        # decoding Pose
        res["pose"] = self.decoder_pose(enc["e_pose"])

        return res
