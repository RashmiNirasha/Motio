#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import torch.nn as nn


class HandHead(nn.Module):
    NUM_CLASS = {
        "object": 1,
        "manipulation": 5,
        "full": 10,
    }

    def __init__(self, cfg, feat_dim=None, hidden_dim=None):
        super().__init__()

        self.tasks = cfg.task.split("_")
        self.single_hand = True if cfg.input_type == "hand" else False

        if hidden_dim is None:
            hidden_dim = cfg.head.hidden_dim

        if feat_dim is None:
            feat_dim = cfg.feat_dim

        dropout_ratio = cfg.head.dropout_ratio
        n_layers = cfg.head.n_layers

        # Define dropout if needed
        self.dropout = nn.Dropout(p=dropout_ratio) if dropout_ratio != 0 else nn.Identity()

        # Classification heads
        for task in self.tasks:
            assert task in self.NUM_CLASS, f"set number of classes in NUM_CLASS for task {task}"
            num_class = self._get_num_class(task)
            if self.single_hand:
                # Single hand: one classifier per task
                self.add_module(
                    f"cls_{task}", self.get_mlp_head(n_layers, feat_dim, hidden_dim, num_class)
                )
            else:
                # Dual hand: separate classifiers for left and right hands
                self.add_module(
                    f"cls_left_{task}",
                    self.get_mlp_head(n_layers, feat_dim, hidden_dim, num_class),
                )
                self.add_module(
                    f"cls_right_{task}",
                    self.get_mlp_head(n_layers, feat_dim, hidden_dim, num_class),
                )

    def _get_num_class(self, task):
        num_class = self.NUM_CLASS.get(task)
        if num_class is None:
            raise ValueError(
                f"Task '{task}' not found in NUM_CLASS. Available: {list(self.NUM_CLASS.keys())}"
            )
        return num_class

    @staticmethod
    def get_mlp_head(n_layers, input_dim, hidden_dim, num_class):
        if n_layers == 1:
            return nn.Linear(input_dim, num_class)

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, num_class))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of the hand head.

        Args:
            x (torch.Tensor): Input features [batch_size, feat_dim]

        Returns:
            dict: Dictionary containing predictions for each task and hand
        """
        res = {}

        for task in self.tasks:
            if self.single_hand:
                # Single hand predictions
                cls = getattr(self, f"cls_{task}")
                res[f"pred_{task}"] = cls(self.dropout(x))
            else:
                # Dual hand predictions
                cls_left = getattr(self, f"cls_left_{task}")
                cls_right = getattr(self, f"cls_right_{task}")
                res[f"left_pred_{task}"] = cls_left(self.dropout(x))
                res[f"right_pred_{task}"] = cls_right(self.dropout(x))

        return res
