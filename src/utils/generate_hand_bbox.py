#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#


from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt


class HandBBoxGenerator:
    """Hand bounding box generator."""

    def __init__(self, extend_factor: float = 0.5, ratio_body: float = 0.4):
        """Initialize the hand bounding box generator.

        Args:
            extend_factor (float): Factor to extend wrist
            ratio_body (float): Ratio of body size to use for hand box size
        """
        self.extend_factor = extend_factor
        self.ratio_body = ratio_body

    @staticmethod
    def is_person_invalid(bbox: Union[List[float], npt.NDArray[np.float64]]) -> bool:
        """Check if a person bounding box is invalid.
        Args:
            bbox (Union[List[float], npt.NDArray[np.float64]]): Person bounding box coordinates

        Returns:
            bool: True if the bounding box is invalid, False otherwise
        """
        if isinstance(bbox, list):
            bbox = np.array(bbox, dtype=np.float64)

        # Check for missing detection
        if np.all(bbox == -1):
            return True

        # Validate box dimensions
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return True

        return False

    def compute_hand_bbox(
        self,
        image_shape: Tuple[int, int],
        person_bbox: npt.NDArray[np.float64],
        keypoints: npt.NDArray[np.float64],
        elbow_idx: int,
        wrist_idx: int,
    ) -> npt.NDArray[np.float64]:
        """Compute hand bounding box from person detection and keypoints.

        This method generates a pseudo hand bounding box by:
        1. Extending the wrist position
        2. Using a fraction of the person's body size as the hand box size (the head bounding may be a better option)
        3. Ensuring the box stays within image boundaries

        Args:
            image_shape (Tuple[int, int]): Image dimensions (height, width)
            person_bbox (npt.NDArray[np.float64]): Person bounding box coordinates
            keypoints (npt.NDArray[np.float64]): Array of keypoint coordinates
            elbow_idx (int): Index of elbow keypoint
            wrist_idx (int): Index of wrist keypoint

        Returns:
            npt.NDArray[np.float64]: Hand bounding box coordinates [x1, y1, x2, y2] or
                [-1, -1, -1, -1] if invalid
        """
        # Return invalid box if person detection is invalid
        if self.is_person_invalid(person_bbox):
            return np.array([-1, -1, -1, -1], dtype=np.float64)

        # Get image dimensions
        h, w = image_shape

        # Compute extended wrist position as center
        wrist_position = keypoints[wrist_idx, :2] + self.extend_factor * (
            keypoints[wrist_idx, :2] - keypoints[elbow_idx, :2]
        )
        center_x, center_y = wrist_position

        # Compute hand box size based on person body size
        body_width = person_bbox[2] - person_bbox[0]
        body_height = person_bbox[3] - person_bbox[1]
        half_dim = (self.ratio_body * min(body_width, body_height)) / 2

        # Generate hand bounding box with image boundary constraints
        hand_bbox = np.array(
            [
                max(0, center_x - half_dim),  # x1
                max(0, center_y - half_dim),  # y1
                min(w, center_x + half_dim),  # x2
                min(h, center_y + half_dim),  # y2
            ],
            dtype=np.float64,
        )

        # Validate box dimensions
        if hand_bbox[2] <= hand_bbox[0] or hand_bbox[3] <= hand_bbox[1]:
            return np.array([-1, -1, -1, -1], dtype=np.float64)

        return hand_bbox
