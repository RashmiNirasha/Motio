#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

import os

import joblib
import pandas as pd
import torch

from src.evaluation.frame_eval import FrameEvaluation
from src.evaluation.segmental_eval import SegmentalEvaluation
from src.utils import get_pylogger

log = get_pylogger(__name__)

# --------------------------------------------------------------
# Task Definitions
# --------------------------------------------------------------
TASKS = {
    # Object in Hand (OiH) task
    "object": {"idx_to_class": {0: "non_oih", 1: "oih"}, "labels": [0, 1]},
    # Manipulation Stages (ManiS) task
    "manipulation": {
        "idx_to_class": {0: "background", 1: "grasp", 2: "hold", 3: "operate", 4: "release"},
        "labels": [0, 1, 2, 3, 4],
    },
    # Full action classification task
    "full": {
        "idx_to_class": {
            0: "background",
            1: "grasp",
            2: "hold",
            3: "operate",
            4: "release",
            5: "give",
            6: "put",
            7: "drop",
            8: "throw",
            9: "point",
        },
        "labels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
}

# --------------------------------------------------------------
# ChildPlay-Hand Evaluator
# --------------------------------------------------------------


class CP_evaluator:
    """Main evaluator for the ChildPlay-Hand.
    Args:
        cfg: Configuration object containing evaluation parameters
        split (str): Data split to evaluate ('train', 'val', or 'test')
    """

    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.tasks = cfg.task.split("_")
        self.input_type = cfg.input_type
        self.task_info = self._set_task_labels()

        # Set up file paths for results storage
        eval_name = cfg.eval_name or "main"
        self.save_root = f"{self.cfg.storage_folder}/eval_{eval_name}"
        self.pred_file_path = f"{self.save_root}/{split}_preds.csv"
        os.makedirs(self.save_root, exist_ok=True)

        # Prepare CSV columns for storing results
        self.columns = ["clip", "frame", "pid", "hand"]
        self.columns += [f"{task}_{suffix}" for task in self.tasks for suffix in ("pred", "gt")]
        self.pred_rows = []

        # Initialize evaluators for frame and segment metrics
        self.frame_evaluator = FrameEvaluation(self.task_info)
        self.seg_evaluator = SegmentalEvaluation(self.task_info)

    def _set_task_labels(self):
        """Get task-specific label mappings.

        Returns:
            dict: Dictionary containing label mappings for each task
        """
        return TASKS

    def reset(self):
        """Reset stored predictions."""
        self.pred_rows = []

    def load_from_csv(self, pred_file_path):
        """Load predictions from CSV file.

        Args:
            pred_file_path (str): Path to the CSV file containing predictions

        Returns:
            dict: Dictionary containing predictions and ground truth for each task
        """
        res_dict = {f"{task}_{suffix}": [] for task in self.tasks for suffix in ("pred", "gt")}
        df = pd.read_csv(pred_file_path)
        df = df[self.columns].groupby(["clip", "pid", "hand"])

        for _, group in df:
            group = group.sort_values(by="frame")
            for task in self.tasks:
                res_dict[f"{task}_pred"].append(group[f"{task}_pred"].tolist())
                res_dict[f"{task}_gt"].append(group[f"{task}_gt"].tolist())

        return res_dict

    @staticmethod
    def logit_to_label(pred):
        """Convert logits to class labels.

        Args:
            pred (torch.Tensor): Model predictions (logits)

        Returns:
            int: Predicted class label
        """
        return (
            (pred.sigmoid() > 0.5).int().item()
            if pred.shape[0] == 1
            else pred.softmax(-1).argmax().item()
        )

    def store_results_sequence(self, input_data, input_meta, output):
        """Store intermediate predictions for segmentation task.

        Args:
            input_data (dict): Input data dictionary
            input_meta (dict): Input metadata dictionary
            output (dict): Model output dictionary
        """
        BS = next(v.shape[0] for v in input_data.values() if v is not None)
        assert BS == 1
        assert self.input_type == "hand"
        bid = 0  # one hand sequence

        # Parse instance information
        ins_name = input_meta["instance_name"][bid][0]  # {clip_name}/{pid}/{hand}
        clip_name = ins_name.split("/")[0]
        pid = ins_name.split("/")[1]
        hand = ins_name.split("/")[2]
        f_names = input_meta["frame_names"][bid]
        T_ = len(f_names)

        # Get predictions and ground truth
        pred = output["pred"][bid, :T_, :]  # (T_, nC)
        pred = torch.argmax(pred, dim=1)  # (T_)
        gt_action = input_data["labels"][bid, :T_][:T_]  # (T_)
        mask = input_data["mask"][bid, 0, :T_]  # (T_)

        # Store results for each frame
        for ix, f in enumerate(f_names):
            if mask[ix].item() != 0:
                result = [clip_name, f.item(), pid, hand, pred[ix].item(), gt_action[ix].item()]
                self.pred_rows.append(result)

    def store_results_batch(self, input_data, output):
        """Store intermediate predictions for recognition task.

        Args:
            input_data (dict): Input data dictionary
            output (dict): Model output dictionary
        """
        BS = next(v.shape[0] for v in input_data.values() if v is not None)

        for bid in range(BS):
            ins_name = input_data["instance_name"][bid]

            # Handle different input types (body vs hand)
            if self.input_type == "body":
                clip_name, key_frame, pid = ins_name.split("/")
                hands = ["left", "right"]
                prefix = lambda x: f"{x}_"
            else:
                clip_name, key_frame, pid, hand = ins_name.split("/")
                hands = [hand]
                prefix = lambda x: ""

            # Store results for each hand
            for hand in hands:
                row = [clip_name, key_frame, pid, hand]
                for task in self.tasks:
                    pred = self.logit_to_label(output[f"{prefix(hand)}pred_{task}"][bid])
                    gt = input_data[f"{prefix(hand)}{task}_task"][bid].item()
                    row += [pred, gt]
                self.pred_rows.append(row)

    def run_evaluation(self, labels_dict):
        """Run frame and segment-level evaluation.

        Args:
            labels_dict (dict): Dictionary containing predictions and ground truth

        Returns:
            dict: Dictionary containing evaluation metrics for each task and level
        """
        return {
            task: {
                "frame": self.frame_evaluator(
                    labels_dict[f"{task}_pred"], labels_dict[f"{task}_gt"], task
                ),
                "segment": self.seg_evaluator(
                    labels_dict[f"{task}_pred"], labels_dict[f"{task}_gt"], task
                ),
            }
            for task in self.tasks
        }

    def compute_metrics_from_csv(self, pred_file_path):
        """Compute metrics directly from CSV file.

        Args:
            pred_file_path (str): Path to the CSV file containing predictions

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Load from CSV and evaluate
        res_dict = self.load_from_csv(pred_file_path)
        results = self.run_evaluation(res_dict)

        return results

    def compute_metrics(self):
        """Compute and save evaluation metrics.

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Save predictions to CSV
        pd.DataFrame(self.pred_rows, columns=self.columns).to_csv(self.pred_file_path, index=False)
        self.reset()
        # Load from CSV and evaluate
        res_dict = self.load_from_csv(self.pred_file_path)
        results = self.run_evaluation(res_dict)

        # Save results for test split
        if self.split == "test":
            joblib.dump(results, f"{self.save_root}/eval_{self.split}.pkl")

        return results
