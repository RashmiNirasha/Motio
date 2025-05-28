#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# --------------------------------------------------------------
# Frame-wise Evaluation
# --------------------------------------------------------------


class FrameEvaluation:
    """Frame-level evaluation.

    Args:
        tasks_info (dict): Dictionary containing task-specific information
            including label mappings and configurations
    """

    def __init__(self, tasks_info):
        self.tasks_info = tasks_info

    def __call__(self, pred_videos, gt_videos, task):
        """Evaluate frame-level predictions.

        Args:
            pred_videos (List[List]): Predicted labels for each frame in videos
            gt_videos (List[List]): Ground truth labels for each frame in videos
            task (str): Task type ('manipulation' or 'object')

        Returns:
            dict: Dictionary containing frame-level evaluation metrics:
                - accuracy: Overall accuracy
                - precision: Precision score (binary or macro)
                - recall: Recall score (binary or macro)
                - f1: F1 score (binary or macro)
                - precision_class: Per-class precision (multi-class only)
                - recall_class: Per-class recall (multi-class only)
                - f1_class: Per-class F1 score (multi-class only)
                - confusion_frame_matrix: Confusion matrix
                - classification_report: Detailed classification report
        """
        # Get task-specific labels
        self.labels = self.tasks_info[task]["labels"]

        # Combine predictions and ground truth from all videos
        all_pred, all_gt = self.combine_labels(pred_videos, gt_videos)

        # Compute overall accuracy
        metrics = {"accuracy": accuracy_score(all_gt, all_pred) * 100}

        # Add task-specific metrics
        if task == "object":
            self._add_binary_metrics(metrics, all_gt, all_pred)
        else:
            self._add_macro_metrics(metrics, all_gt, all_pred)
            self._add_class_metrics(metrics, all_gt, all_pred)

        # Add confusion matrix and classification report
        metrics["confusion_frame_matrix"] = confusion_matrix(
            all_gt, all_pred, labels=self.labels if task != "object" else None
        )
        metrics["classification_report"] = classification_report(
            all_gt, all_pred, labels=self.labels if task != "object" else None, output_dict=True
        )

        return metrics

    @staticmethod
    def combine_labels(pred_videos, gt_videos):
        """Combine predictions and ground truth from all videos.

        Args:
            pred_videos (List[List]): Predicted labels for each frame in videos
            gt_videos (List[List]): Ground truth labels for each frame in videos

        Returns:
            tuple: Combined lists of predictions and ground truth labels
        """
        all_pred, all_gt = [], []
        for pred, gt in zip(pred_videos, gt_videos):
            all_pred.extend(pred)
            all_gt.extend(gt)
        return all_pred, all_gt

    def _add_macro_metrics(self, metrics, all_gt, all_pred):
        """Add macro-averaged metrics for multi-class classification.

        Args:
            metrics (dict): Dictionary to update with macro metrics
            all_gt (List): Combined ground truth labels
            all_pred (List): Combined predicted labels
        """
        metrics.update(
            {
                "precision": precision_score(all_gt, all_pred, average="macro", labels=self.labels)
                * 100,
                "recall": recall_score(all_gt, all_pred, average="macro", labels=self.labels)
                * 100,
                "f1": f1_score(all_gt, all_pred, average="macro", labels=self.labels) * 100,
            }
        )

    def _add_class_metrics(self, metrics, all_gt, all_pred):
        """Add per-class metrics for multi-class classification.

        Args:
            metrics (dict): Dictionary to update with per-class metrics
            all_gt (List): Combined ground truth labels
            all_pred (List): Combined predicted labels
        """
        metrics.update(
            {
                "precision_class": precision_score(all_gt, all_pred, average=None),
                "recall_class": recall_score(all_gt, all_pred, average=None),
                "f1_class": f1_score(all_gt, all_pred, average=None),
            }
        )

    def _add_binary_metrics(self, metrics, all_gt, all_pred):
        """Add binary classification metrics.

        Args:
            metrics (dict): Dictionary to update with binary metrics
            all_gt (List): Combined ground truth labels
            all_pred (List): Combined predicted labels
        """
        metrics.update(
            {
                "precision": precision_score(all_gt, all_pred, average="binary") * 100,
                "recall": recall_score(all_gt, all_pred, average="binary") * 100,
                "f1": f1_score(all_gt, all_pred, average="binary") * 100,
            }
        )
