#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

# https://github.com/yabufarha/ms-tcn/blob/master/model.py
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------
class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps))
                for i in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


# -------------------------------------------------------------
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


# -------------------------------------------------------------
class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList(
            [
                copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes))
                for s in range(num_stages - 1)
            ]
        )

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


# -------------------------------------------------------------
NUM_CLASS = {"object": 2, "manipulation": 5, "full": 10}


# -------------------------------------------------------------
# MSTCN
# -------------------------------------------------------------
class MSTCN(nn.Module):
    def __init__(self, args):
        super(MSTCN, self).__init__()
        self.input_dim = args.input_dim
        self.mstcn = MultiStageModel(
            args.num_stages,
            args.num_layers,
            args.num_f_maps,
            args.input_dim,
            NUM_CLASS[args.task],
        )

    def forward(self, x, return_last_stage=True):
        outputs = {}
        feats = x["feats"][:, : self.input_dim, :]
        outputs["pred"] = self.mstcn(feats, x["mask"])

        if return_last_stage:
            outputs_ls = {}
            outputs_ls["pred"] = outputs["pred"][-1].permute(0, 2, 1).contiguous()
            return outputs, outputs_ls
        return outputs


# -------------------------------------------------------------
# MSTCN Loss
# -------------------------------------------------------------
class MSTCN_LOSS:
    def __init__(self, args):
        self.num_classes = NUM_CLASS[args.task]
        self.truncated_mse = args.truncated_mse

    def compute_loss(self, predictions, batch_target, mask):
        loss = 0
        for p in predictions:
            # CE
            loss += F.cross_entropy(
                p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                batch_target.view(-1),
                ignore_index=-100,
            )
            # Truncated MSE
            if self.truncated_mse:
                loss += 0.15 * torch.mean(
                    torch.clamp(
                        F.mse_loss(
                            F.log_softmax(p[:, :, 1:], dim=1),
                            F.log_softmax(p.detach()[:, :, :-1], dim=1),
                            reduction="none",
                        ),
                        min=0,
                        max=16,
                    )
                    * mask[:, :, 1:]
                )
        return loss

    def __call__(self, output, batch):
        return {"action": self.compute_loss(output["pred"], batch["labels"], batch["mask"])}
