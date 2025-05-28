#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import warnings

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")


# -------------------------------------------------------------
class HandLoss:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tasks = cfg.task.split("_")
        self.modality = [m.lower() for m in cfg.modality]
        self.parse_pred = lambda pred, mod: pred if len(self.modality) == 1 else pred[mod]

    @staticmethod
    def _get_loss_func(task):
        if task == "object":
            return lambda pred, gt: F.binary_cross_entropy_with_logits(pred.view(-1), gt.float())
        elif task == "manipulation":
            return F.cross_entropy
        elif task == "full":
            return F.cross_entropy
        else:
            raise NotImplementedError("Unsupported task")

    def compute_loss(self, preds, ground_truth, task):
        loss_func = self._get_loss_func(task)
        loss = torch.tensor(0.0, device="cuda")

        if f"left_pred_{task}" in preds:
            loss += 0.5 * loss_func(preds[f"left_pred_{task}"], ground_truth[f"left_{task}_task"])
        if f"right_pred_{task}" in preds:
            loss += 0.5 * loss_func(
                preds[f"right_pred_{task}"], ground_truth[f"right_{task}_task"]
            )
        if f"pred_{task}" in preds:
            loss += loss_func(preds[f"pred_{task}"], ground_truth[f"{task}_task"])

        return loss

    def __call__(self, preds, ground_truth):
        return {
            f"{mod}_{task}": self.compute_loss(self.parse_pred(preds, mod), ground_truth, task)
            for mod in self.modality
            for task in self.tasks
        }
