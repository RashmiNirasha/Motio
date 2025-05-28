#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

"""
Partially based on the MS-TCN implementation:
https://github.com/yabufarha/ms-tcn/blob/master/eval.py
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.utils import get_pylogger

log = get_pylogger(__name__)


# --------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------
def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def label_segments(frames: list, bg_class: list = []) -> tuple:
    """Extract segments with start and end times for each label.

    Args:
        frames (list): List of frame labels
        bg_class (list, optional): List of background class labels. Defaults to [].

    Returns:
        tuple: Lists of labels, start times, and end times
    """
    labels, starts, ends = [], [], []
    last_label = frames[0]

    if last_label not in bg_class:
        labels.append(last_label)
        starts.append(0)

    for i, label in enumerate(frames):
        if label != last_label:
            if label not in bg_class:
                labels.append(label)
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = label

    if last_label not in bg_class:
        ends.append(i)

    return labels, starts, ends


def levenstein_distance(p: list, y: list, norm: bool = False) -> float:
    """Compute Levenshtein distance (edit distance) between two sequences.

    Args:
        p (list): First sequence
        y (list): Second sequence
        norm (bool, optional): Whether to normalize the distance. Defaults to False.

    Returns:
        float: Levenshtein distance between sequences
    """
    m, n = len(p), len(y)
    D = np.zeros((m + 1, n + 1))

    for i in range(m + 1):
        D[i, 0] = i
    for i in range(n + 1):
        D[0, i] = i

    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + 1)

    if norm:
        return (1 - D[m, n] / max(m, n)) * 100
    else:
        return D[m, n]


def edit_score(pred: list, gt: list, norm: bool = True, bg_class: list = ["background"]) -> float:
    """Compute edit score between prediction and ground truth sequences.

    Args:
        pred (list): Predicted sequence
        gt (list): Ground truth sequence
        norm (bool, optional): Whether to normalize the score. Defaults to True.
        bg_class (list, optional): List of background class labels. Defaults to ["background"].

    Returns:
        float: Edit score between sequences
    """
    P, _, _ = label_segments(pred, bg_class)
    Y, _, _ = label_segments(gt, bg_class)
    return levenstein_distance(P, Y, norm)


# --------------------------------------------------------------
# Matching Functions
# --------------------------------------------------------------


def segment_frames(frames: list, class_label: int = None) -> list:
    """Segment consecutive frames with the same label.

    Args:
        frames (list): List of frame labels
        class_label (int, optional): Specific class label to filter segments. Defaults to None.

    Returns:
        list: List of (start, end, label) tuples for each segment
    """
    segments = []
    start = 0
    current_label = frames[0]

    for i, label in enumerate(frames[1:], 1):
        if label != current_label:
            segments.append((start, i - 1, current_label))
            start = i
            current_label = label

    segments.append((start, len(frames) - 1, current_label))

    if class_label is not None:
        return [seg for seg in segments if seg[2] == class_label]

    return segments


def overlap(det: tuple, gt: tuple) -> float:
    """Calculate overlap between two segments.

    Args:
        det (tuple): Detection segment (start, end)
        gt (tuple): Ground truth segment (start, end)

    Returns:
        float: F1 score of segment overlap
    """
    start_det, end_det = det
    start_gt, end_gt = gt
    intersection = len(
        set(range(start_det, end_det + 1)).intersection(range(start_gt, end_gt + 1))
    )

    if intersection:
        p = intersection / (end_det - start_det + 1)
        r = intersection / (end_gt - start_gt + 1)
        return 2 * p * r / (p + r)

    return 0


def cost_matrix(detections: list, ground_truths: list) -> np.ndarray:
    """Create cost matrix for matching detections with ground truths.

    Args:
        detections (list): List of detection segments
        ground_truths (list): List of ground truth segments

    Returns:
        np.ndarray: Cost matrix for matching
    """
    matrix = np.zeros((len(detections), len(ground_truths)))

    for i, det in enumerate(detections):
        for j, gt in enumerate(ground_truths):
            if det[2] == gt[2]:
                matrix[i, j] = 1 - overlap((det[0], det[1]), (gt[0], gt[1]))
            else:
                matrix[i, j] = 2  # High cost for mismatched classes

    return matrix


def match_segments(detections: list, ground_truths: list) -> tuple:
    """Perform matching using the Hungarian algorithm.

    Args:
        detections (list): List of detection segments
        ground_truths (list): List of ground truth segments

    Returns:
        tuple: Cost matrix and matching indices
    """
    matrix = cost_matrix(detections, ground_truths)
    row_ind, col_ind = linear_sum_assignment(matrix)
    return matrix, row_ind, col_ind


def compute_metrics(
    detections: list,
    ground_truths: list,
    row_ind: list,
    col_ind: list,
    matrix: np.ndarray,
    labels: list,
) -> tuple:
    """Calculate precision, recall, and F1 score for each class.

    Args:
        detections (list): List of detection segments
        ground_truths (list): List of ground truth segments
        row_ind (list): Row indices from matching
        col_ind (list): Column indices from matching
        matrix (np.ndarray): Cost matrix
        labels (list): List of class labels

    Returns:
        tuple: True positives, false positives, and false negatives for each class
    """
    tp, fp, fn = {l: 0 for l in labels}, {l: 0 for l in labels}, {l: 0 for l in labels}

    used_gt, used_pred = [], []
    for i in range(len(row_ind)):
        if matrix[row_ind[i], col_ind[i]] < 1:
            tp[detections[row_ind[i]][2]] += 1
            used_gt.append(col_ind[i])
            used_pred.append(row_ind[i])
        else:
            fp[detections[row_ind[i]][2]] += 1
            used_pred.append(row_ind[i])

    for i in range(len(detections)):
        if i not in used_pred:
            fp[detections[i][2]] += 1

    for i in range(len(ground_truths)):
        if i not in used_gt:
            fn[ground_truths[i][2]] += 1

    return tp, fp, fn


def run_matching(preds: list, gts: list, labels: list, bg_class: list = []) -> dict:
    """Run matching and compute evaluation metrics.

    Args:
        preds (list): List of predicted sequences
        gts (list): List of ground truth sequences
        labels (list): List of class labels
        bg_class (list, optional): List of background class labels. Defaults to [].

    Returns:
        dict: Dictionary containing evaluation metrics:
            - edit_score: Average edit score
            - precision: Macro-averaged precision
            - recall: Macro-averaged recall
            - f1: Macro-averaged F1 score
    """
    results = {}
    tp, fp, fn = {l: 0 for l in labels}, {l: 0 for l in labels}, {l: 0 for l in labels}
    precision, recall, f1 = {}, {}, {}
    edit_sum = 0.0

    for pred, gt in zip(preds, gts):
        edit_sum += edit_score(pred, gt, norm=True, bg_class=bg_class)
        pred_segments = segment_frames(pred)
        gt_segments = segment_frames(gt)
        cost_matrix, row_ind, col_ind = match_segments(pred_segments, gt_segments)
        tp_cls, fp_cls, fn_cls = compute_metrics(
            pred_segments, gt_segments, row_ind, col_ind, cost_matrix, labels
        )

        for cls in labels:
            tp[cls] += tp_cls[cls]
            fp[cls] += fp_cls[cls]
            fn[cls] += fn_cls[cls]

    for cls in labels:
        precision[cls] = safe_div(tp[cls], tp[cls] + fp[cls])
        recall[cls] = safe_div(tp[cls], tp[cls] + fn[cls])
        f1[cls] = safe_div(2 * precision[cls] * recall[cls], precision[cls] + recall[cls])

    results["edit_score"] = edit_sum / len(gts)
    results["precision"] = safe_div(sum(precision.values()), len(labels)) * 100.0  # macro
    results["recall"] = safe_div(sum(recall.values()), len(labels)) * 100.0  # macro
    results["f1"] = safe_div(sum(f1.values()), len(labels)) * 100.0  # macro

    return results


# --------------------------------------------------------------
# Segmental Evaluation
# --------------------------------------------------------------


class SegmentalEvaluation:
    """Segmental evaluation.
    Args:
        tasks_info (dict): Dictionary containing task-specific information
        bg_class (list, optional): List of background class labels. Defaults to [].
    """

    def __init__(self, tasks_info, bg_class=[]):
        self.tasks_info = tasks_info
        self.bg_class = bg_class

    def __call__(self, pred_videos: list, gt_videos: list, task: str) -> dict:
        """Evaluate segment-level predictions.

        Args:
            pred_videos (list): List of predicted sequences
            gt_videos (list): List of ground truth sequences
            task (str): Task type

        Returns:
            dict: Dictionary containing segment-level evaluation metrics
        """
        labels = self.tasks_info[task]["labels"]
        return run_matching(pred_videos, gt_videos, labels, bg_class=self.bg_class)
