#
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Arya Farkhondeh <arya.farkhondeh@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only
#

# Mostley adopted from PYSKL: https://github.com/kennymckormick/pyskl

import os
from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from PIL import Image
from torch.nn.modules.utils import _pair


# --------------------------------------------------------------
# Generic transforms
# --------------------------------------------------------------
def get_frame_path(root, clip, frame_id):
    video_id, interval = clip.replace("-downsample", "").rsplit("_", 1)
    offset = int(interval.split("-")[0])
    img_name = f"{video_id}_{offset + frame_id - 1}.jpg"
    path = os.path.join("images", clip, img_name)
    frame_path = os.path.join(root, path)

    return frame_path


class ClipSampler:
    """
    ClipSampler: a generic multimodal clip sampler
    """

    @staticmethod
    def generate_indicies(window_size, sampling_rate, key_index, max_clip):
        """
        A generic function to generate indices of a sample

        window_size: effective window size
        sampling_rate: rate for subsampling
        key_index: index of middle frame
        max_clip: length of the video
        """
        if window_size > 1:
            half_w = window_size // 2
            indices = list(
                range(key_index - half_w, min(key_index + half_w, max_clip), sampling_rate)
            )
            indices = [num for num in indices if num >= 0]
        else:
            indices = [key_index]
        return indices

    def __call__(self, sample):
        key = sample["key"]
        frame_numbs = sample["frame_numbs"]
        window_size = sample["window_size"]

        key_frame = key[1]
        key_index = frame_numbs.index(key_frame)
        max_clip = len(frame_numbs)

        sample["key_index"] = key_index

        # sample global indices for general use
        sample["global_indicies"] = self.generate_indicies(window_size, 1, key_index, max_clip)
        assert len(sample["global_indicies"]) <= window_size

        # sample modality-specific indicies
        for mod in sample["modality"]:
            sampling_rate = sample[mod]["sampling_rate"]
            window_size = sample[mod]["window_size"]
            sample[mod]["frame_inds"] = self.generate_indicies(
                window_size, sampling_rate, key_index, max_clip
            )
            assert len(sample[mod]["frame_inds"]) <= window_size

        return sample


# --------------------------------------------------------------
# Pose transforms
# --------------------------------------------------------------
class PoseDecode:
    @staticmethod
    def _load_keypoint(kp, frame_inds):
        return kp[:, frame_inds].astype(np.float32)

    @staticmethod
    def _load_keypoint_score(kpscore, frame_inds):
        return kpscore[:, frame_inds].astype(np.float32)

    @staticmethod
    def _pad_pose(results):
        n_keypoints = results["n_keypoints"]
        clip_len = results["Pose"]["clip_len"]
        keypoint = np.zeros((1, clip_len, n_keypoints, 2), dtype=np.float32)
        keypoint_score = np.zeros((1, clip_len, n_keypoints), dtype=np.float32)

        index_keyidx = results["Pose"]["frame_inds"].index(results["key_index"])
        mid_point = clip_len // 2
        st_offset = mid_point - index_keyidx
        end_offset = st_offset + len(results["Pose"]["frame_inds"])

        keypoint[:, st_offset:end_offset] = results["keypoint"]
        keypoint_score[:, st_offset:end_offset] = results["keypoint_score"]

        results["keypoint"] = keypoint
        results["keypoint_score"] = keypoint_score

        return results

    def __call__(self, results):
        assert all(key in results for key in ["Pose", "keypoint", "keypoint_score"])

        frame_inds = results["Pose"]["frame_inds"]

        results["keypoint"] = self._load_keypoint(results["keypoint"], frame_inds)  # 1, nf, nk, 2
        results["keypoint_score"] = self._load_keypoint_score(
            results["keypoint_score"], frame_inds
        )  # 1, nf, nk, 2

        if len(frame_inds) != results["Pose"]["clip_len"]:
            results = self._pad_pose(results)

        return results


class BBoxDecode:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @staticmethod
    def _load_bbox(bbox, frame_inds):
        return bbox[:, frame_inds].astype(np.float32)

    @staticmethod
    def _pad_bbox(results):
        clip_len = results["clip_len"]
        bbox = np.zeros((1, clip_len, 4), dtype=np.float32) - 1.0

        index_keyidx = results["global_indicies"].index(results["key_index"])
        mid_point = clip_len // 2
        st_offset = mid_point - index_keyidx
        end_offset = st_offset + len(results["global_indicies"])

        bbox[:, st_offset:end_offset] = results["bbox"]
        results["bbox"] = bbox

        return results

    def __call__(self, results):
        # Read Bbox
        results["bbox"] = self._load_bbox(results["bbox"], results["global_indicies"])
        if len(results["global_indicies"]) != results["clip_len"]:
            results = self._pad_bbox(results)

        return results


EPS = 1e-3


class GeneratePoseTarget:
    """Generate pseudo heatmaps based on joint coordinates and confidence.
    see: https://github.com/kennymckormick/pyskl/tree/main/pyskl/datasets/pipelines
    """

    def __init__(
        self,
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        skeletons=(
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (0, 5),
            (5, 7),
            (7, 9),
            (0, 6),
            (6, 8),
            (8, 10),
            (5, 11),
            (11, 13),
            (13, 15),
            (6, 12),
            (12, 14),
            (14, 16),
            (11, 12),
        ),
        double=False,
        left_kp=(1, 3, 5, 7, 9, 11, 13, 15),
        right_kp=(2, 4, 6, 8, 10, 12, 14, 16),
        left_limb=(0, 2, 4, 5, 6, 10, 11, 12),
        right_limb=(1, 3, 7, 8, 9, 13, 14, 15),
        scaling=1.0,
    ):
        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.double = double

        assert (
            self.with_kp + self.with_limb == 1
        ), 'One of "with_limb" and "with_kp" should be set as True.'
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.skeletons = skeletons
        self.left_limb = left_limb
        self.right_limb = right_limb
        self.scaling = scaling

    def generate_a_heatmap(self, arr, centers, max_values):
        """Generate pseudo heatmap for one keypoint in one frame.

        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
            centers (np.ndarray): The coordinates of corresponding keypoints (of multiple persons). Shape: M * 2.
            max_values (np.ndarray): The max values of each keypoint. Shape: M.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        sigma = self.sigma
        img_h, img_w = arr.shape

        for center, max_value in zip(centers, max_values):
            if max_value < EPS:
                continue

            mu_x, mu_y = center[0], center[1]
            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma**2)
            patch = patch * max_value
            arr[st_y:ed_y, st_x:ed_x] = np.maximum(arr[st_y:ed_y, st_x:ed_x], patch)

    def generate_a_limb_heatmap(self, arr, starts, ends, start_values, end_values):
        """Generate pseudo heatmap for one limb in one frame.

        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
            starts (np.ndarray): The coordinates of one keypoint in the corresponding limbs. Shape: M * 2.
            ends (np.ndarray): The coordinates of the other keypoint in the corresponding limbs. Shape: M * 2.
            start_values (np.ndarray): The max values of one keypoint in the corresponding limbs. Shape: M.
            end_values (np.ndarray): The max values of the other keypoint in the corresponding limbs. Shape: M.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        sigma = self.sigma
        img_h, img_w = arr.shape

        for start, end, start_value, end_value in zip(starts, ends, start_values, end_values):
            value_coeff = min(start_value, end_value)
            if value_coeff < EPS:
                continue

            min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
            min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

            min_x = max(int(min_x - 3 * sigma), 0)
            max_x = min(int(max_x + 3 * sigma) + 1, img_w)
            min_y = max(int(min_y - 3 * sigma), 0)
            max_y = min(int(max_y + 3 * sigma) + 1, img_h)

            x = np.arange(min_x, max_x, 1, np.float32)
            y = np.arange(min_y, max_y, 1, np.float32)

            if not (len(x) and len(y)):
                continue

            y = y[:, None]
            x_0 = np.zeros_like(x)
            y_0 = np.zeros_like(y)

            # distance to start keypoints
            d2_start = (x - start[0]) ** 2 + (y - start[1]) ** 2

            # distance to end keypoints
            d2_end = (x - end[0]) ** 2 + (y - end[1]) ** 2

            # the distance between start and end keypoints.
            d2_ab = (start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2

            if d2_ab < 1:
                self.generate_a_heatmap(arr, start[None], start_value[None])
                continue

            coeff = (d2_start - d2_end + d2_ab) / 2.0 / d2_ab

            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate - b_dominate

            position = np.stack([x + y_0, y + x_0], axis=-1)
            projection = start + np.stack([coeff, coeff], axis=-1) * (end - start)
            d2_line = position - projection
            d2_line = d2_line[:, :, 0] ** 2 + d2_line[:, :, 1] ** 2
            d2_seg = a_dominate * d2_start + b_dominate * d2_end + seg_dominate * d2_line

            patch = np.exp(-d2_seg / 2.0 / sigma**2)
            patch = patch * value_coeff

            arr[min_y:max_y, min_x:max_x] = np.maximum(arr[min_y:max_y, min_x:max_x], patch)

    def generate_heatmap(self, arr, kps, max_values):
        """Generate pseudo heatmap for all keypoints and limbs in one frame (if
        needed).

        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: V * img_h * img_w.
            kps (np.ndarray): The coordinates of keypoints in this frame. Shape: M * V * 2.
            max_values (np.ndarray): The confidence score of each keypoint. Shape: M * V.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        if self.with_kp:
            num_kp = kps.shape[1]
            for i in range(num_kp):
                self.generate_a_heatmap(arr[i], kps[:, i], max_values[:, i])

        if self.with_limb:
            for i, limb in enumerate(self.skeletons):
                start_idx, end_idx = limb
                starts = kps[:, start_idx]
                ends = kps[:, end_idx]

                start_values = max_values[:, start_idx]
                end_values = max_values[:, end_idx]
                self.generate_a_limb_heatmap(arr[i], starts, ends, start_values, end_values)

    def gen_an_aug(self, results):
        """Generate pseudo heatmaps for all frames.

        Args:
            results (dict): The dictionary that contains all info of a sample.

        Returns:
            list[np.ndarray]: The generated pseudo heatmaps.
        """

        all_kps = results["keypoint"]
        kp_shape = all_kps.shape

        if "keypoint_score" in results:
            all_kpscores = results["keypoint_score"]
        else:
            all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)

        img_h, img_w = results["img_shape"]

        # scale img_h, img_w and kps
        img_h = int(img_h * self.scaling + 0.5)
        img_w = int(img_w * self.scaling + 0.5)
        all_kps[..., :2] *= self.scaling

        num_frame = kp_shape[1]
        num_c = 0
        if self.with_kp:
            num_c += all_kps.shape[2]
        if self.with_limb:
            num_c += len(self.skeletons)
        ret = np.zeros([num_frame, num_c, img_h, img_w], dtype=np.float32)

        for i in range(num_frame):
            # M, V, C
            kps = all_kps[:, i]
            # M, C
            kpscores = all_kpscores[:, i] if self.use_score else np.ones_like(all_kpscores[:, i])

            self.generate_heatmap(ret[i], kps, kpscores)
        return ret

    def __call__(self, results):
        heatmap = self.gen_an_aug(results)
        key = "heatmap_imgs" if "imgs" in results else "imgs"
        results[key] = heatmap
        return results


def _combine_quadruple(a, b):
    return (a[0] + a[2] * b[0], a[1] + a[3] * b[1], a[2] * b[2], a[3] * b[3])


class PoseCompact_BBox:
    def __init__(self, padding=0.0, threshold=5, hw_ratio=None, allow_imgpad=True):
        self.padding = padding
        self.threshold = threshold
        if hw_ratio is not None:
            hw_ratio = _pair(hw_ratio)

        self.hw_ratio = hw_ratio

        self.allow_imgpad = allow_imgpad
        assert self.padding >= 0

    def _get_box_from_bbox(self, bbox, img_shape):
        h, w = img_shape

        bbox_x1 = bbox[:, :, 0]
        bbox_y1 = bbox[:, :, 1]
        bbox_x2 = bbox[:, :, 2]
        bbox_y2 = bbox[:, :, 3]

        min_x = np.min(bbox_x1[bbox_x1 != -1], initial=np.Inf)
        min_y = np.min(bbox_y1[bbox_y1 != -1], initial=np.Inf)
        max_x = np.max(bbox_x2[bbox_x2 != -1], initial=-np.Inf)
        max_y = np.max(bbox_y2[bbox_y2 != -1], initial=-np.Inf)

        # The compact area is too small
        if max_x - min_x < self.threshold or max_y - min_y < self.threshold:
            return (0, 0, w, h)

        center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        half_width = (max_x - min_x) / 2 * (1 + self.padding)
        half_height = (max_y - min_y) / 2 * (1 + self.padding)

        if self.hw_ratio is not None:
            half_height = max(self.hw_ratio[0] * half_width, half_height)
            half_width = max(1 / self.hw_ratio[1] * half_height, half_width)

        min_x, max_x = center[0] - half_width, center[0] + half_width
        min_y, max_y = center[1] - half_height, center[1] + half_height

        # hot update
        if not self.allow_imgpad:
            min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
            max_x, max_y = int(min(w, max_x)), int(min(h, max_y))
        else:
            min_x, min_y = int(min_x), int(min_y)
            max_x, max_y = int(max_x), int(max_y)

        return (min_x, min_y, max_x, max_y)

    def __call__(self, results):
        img_shape = results["img_shape"]
        h, w = img_shape

        bbox = results["bbox"]  # T x 4: pseudo-hand or gt body bbox
        min_x, min_y, max_x, max_y = self._get_box_from_bbox(bbox, img_shape)

        kp = results["keypoint"]
        kp[np.isnan(kp)] = 0.0
        kp_x, kp_y = kp[..., 0], kp[..., 1]
        kp_x[kp_x != 0] -= min_x
        kp_y[kp_y != 0] -= min_y

        new_shape = (max_y - min_y, max_x - min_x)
        results["img_shape"] = new_shape

        # the order is x, y, w, h (in [0, 1]), a tuple
        crop_quadruple = results.get("crop_quadruple", (0.0, 0.0, 1.0, 1.0))
        new_crop_quadruple = (min_x / w, min_y / h, (max_x - min_x) / w, (max_y - min_y) / h)
        crop_quadruple = _combine_quadruple(crop_quadruple, new_crop_quadruple)
        results["crop_quadruple"] = crop_quadruple

        return results


# --------------------------------------------------------------
# RGB transforms
# --------------------------------------------------------------
class RGBDecode:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @staticmethod
    def _load_bbox(bbox, frame_inds):
        return bbox[:, frame_inds].astype(np.float32)

    @staticmethod
    def _read_img(im, h, w):
        if os.path.exists(im):
            return np.array(Image.open(im))
        return np.zeros((h, w, 3), dtype=np.uint8)

    @staticmethod
    def _read_video(sample):
        vr = sample["video_reader"]
        return vr.get_batch(sample["RGB"]["frame_inds"]).asnumpy()

    @staticmethod
    def _pad_bbox(results):
        clip_len = results["clip_len"]
        bbox = np.zeros((1, clip_len, 4), dtype=np.float32) - 1.0

        index_keyidx = results["global_indicies"].index(results["key_index"])
        mid_point = clip_len // 2
        st_offset = mid_point - index_keyidx
        end_offset = st_offset + len(results["global_indicies"])

        bbox[:, st_offset:end_offset] = results["bbox"]
        results["bbox"] = bbox

        return results

    @staticmethod
    def _pad_imgs(results):
        h, w = results["imgs"][0].shape[:2]
        clip_len = results["RGB"]["clip_len"]
        imgs = np.zeros((clip_len, h, w, 3)) * 0.0

        index_keyidx = results["RGB"]["frame_inds"].index(results["key_index"])
        mid_point = clip_len // 2
        st_offset = mid_point - index_keyidx
        end_offset = st_offset + len(results["RGB"]["frame_inds"])
        imgs[st_offset:end_offset] = np.stack(results["imgs"], axis=0)

        results["imgs"] = [imgs[i] for i in range(imgs.shape[0])]

        return results

    def __call__(self, results):
        # Read RGB
        assert "RGB" in results["modality"]
        img_shape = results["img_shape"]
        h, w = img_shape
        if "video_reader" not in results:
            frame_paths = [
                get_frame_path(results["root_vid"], results["clip"], results["frame_numbs"][ix])
                for ix in results["RGB"]["frame_inds"]
            ]
            imgs = [self._read_img(im, h, w) for im in frame_paths]
        else:
            imgs = self._read_video(results)
            imgs = [imgs[ix] for ix in range(imgs.shape[0])]
        results["imgs"] = imgs
        if len(imgs) != results["RGB"]["clip_len"]:
            results = self._pad_imgs(results)

        # Read Bbox
        results["bbox"] = self._load_bbox(results["bbox"], results["global_indicies"])
        if len(results["global_indicies"]) != results["clip_len"]:
            results = self._pad_bbox(results)

        return results


class MMDecode:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.pose_decode = PoseDecode(**kwargs)
        self.rgb_decode = RGBDecode(**kwargs)

    def __call__(self, results):
        results = self.pose_decode(results)
        results = self.rgb_decode(results)

        return results


class MMCompact_BBox:
    def __init__(self, padding=0.25, threshold=10, hw_ratio=1, allow_imgpad=True):
        self.padding = padding
        self.threshold = threshold
        if hw_ratio is not None:
            hw_ratio = _pair(hw_ratio)
        self.hw_ratio = hw_ratio
        self.allow_imgpad = allow_imgpad
        assert self.padding >= 0

    def _compact_images(self, imgs, img_shape, box):
        h, w = img_shape
        min_x, min_y, max_x, max_y = box
        pad_l, pad_u, pad_r, pad_d = 0, 0, 0, 0
        if min_x < 0:
            pad_l = -min_x
            min_x, max_x = 0, max_x + pad_l
            w += pad_l
        if min_y < 0:
            pad_u = -min_y
            min_y, max_y = 0, max_y + pad_u
            h += pad_u
        if max_x > w:
            pad_r = max_x - w
            w = max_x
        if max_y > h:
            pad_d = max_y - h
            h = max_y

        if pad_l > 0 or pad_r > 0 or pad_u > 0 or pad_d > 0:
            imgs = [np.pad(img, ((pad_u, pad_d), (pad_l, pad_r), (0, 0))) for img in imgs]
        imgs = [img[min_y:max_y, min_x:max_x] for img in imgs]

        return imgs

    def get_actual_bbox(self, input_, img_shape, new_box):
        h, w = img_shape

        bbox_x1 = input_[:, :, 0]
        bbox_y1 = input_[:, :, 1]
        bbox_x2 = input_[:, :, 2]
        bbox_y2 = input_[:, :, 3]
        min_x = np.min(bbox_x1[bbox_x1 != -1], initial=np.Inf)
        min_y = np.min(bbox_y1[bbox_y1 != -1], initial=np.Inf)
        max_x = np.max(bbox_x2[bbox_x2 != -1], initial=-np.Inf)
        max_y = np.max(bbox_y2[bbox_y2 != -1], initial=-np.Inf)

        # The compact area is too small
        if max_x - min_x < self.threshold or max_y - min_y < self.threshold:
            return (0, 0, w, h)

        padding = 0
        center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        half_width = (max_x - min_x) / 2 * (1 + padding)
        half_height = (max_y - min_y) / 2 * (1 + padding)

        min_x, max_x = center[0] - half_width, center[0] + half_width
        min_y, max_y = center[1] - half_height, center[1] + half_height

        min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
        max_x, max_y = int(min(w, max_x)), int(min(h, max_y))

        tight_box = (
            max(0, int(min_x - new_box[0])),
            max(0, int(min_y - new_box[1])),
            min(int(max_x - new_box[0]), new_box[2]),
            min(int(max_y - new_box[1]), new_box[3]),
        )
        return tight_box

    def _get_box_from_bbox(self, bbox, img_shape):
        h, w = img_shape

        bbox_x1 = bbox[:, :, 0]
        bbox_y1 = bbox[:, :, 1]
        bbox_x2 = bbox[:, :, 2]
        bbox_y2 = bbox[:, :, 3]

        min_x = np.min(bbox_x1[bbox_x1 != -1], initial=np.Inf)
        min_y = np.min(bbox_y1[bbox_y1 != -1], initial=np.Inf)
        max_x = np.max(bbox_x2[bbox_x2 != -1], initial=-np.Inf)
        max_y = np.max(bbox_y2[bbox_y2 != -1], initial=-np.Inf)

        # The compact area is too small
        if max_x - min_x < self.threshold or max_y - min_y < self.threshold:
            return (0, 0, w, h)

        center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        half_width = (max_x - min_x) / 2 * (1 + self.padding)
        half_height = (max_y - min_y) / 2 * (1 + self.padding)

        if self.hw_ratio is not None:
            half_height = max(self.hw_ratio[0] * half_width, half_height)
            half_width = max(1 / self.hw_ratio[1] * half_height, half_width)

        min_x, max_x = center[0] - half_width, center[0] + half_width
        min_y, max_y = center[1] - half_height, center[1] + half_height

        if not self.allow_imgpad:
            min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
            max_x, max_y = int(min(w, max_x)), int(min(h, max_y))
        else:
            min_x, min_y = int(min_x), int(min_y)
            max_x, max_y = int(max_x), int(max_y)

        return (min_x, min_y, max_x, max_y)

    def __call__(self, results):
        img_shape = results["img_shape"]

        bbox = results["bbox"]
        min_x, min_y, max_x, max_y = self._get_box_from_bbox(bbox, img_shape)

        if results["input_type"] == "body":
            results["gt_bboxes"] = self.get_actual_bbox(
                bbox, img_shape, (min_x, min_y, max_x, max_y)
            )

        if "Pose" in results["modality"]:
            kp = results["keypoint"]
            kp[np.isnan(kp)] = 0.0
            kp_x, kp_y = kp[..., 0], kp[..., 1]
            kp_x[kp_x != 0] -= min_x
            kp_y[kp_y != 0] -= min_y

        new_shape = (max_y - min_y, max_x - min_x)
        results["img_shape"] = new_shape
        results["imgs"] = self._compact_images(
            results["imgs"], img_shape, (min_x, min_y, max_x, max_y)
        )

        return results


# --------------------------------------------------------------
# Misc
# --------------------------------------------------------------
class Compose:
    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = self.build_from_cfg(transform)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(
                    f"transform must be callable or a dict, " f"but got {type(transform)}"
                )

    @staticmethod
    def build_from_cfg(cfg):
        if "type" not in cfg:
            print(cfg)
            raise ValueError("Configuration dictionary must have a 'type' key.")

        class_type = cfg.pop("type")
        if class_type not in globals():
            raise ValueError(f"Class {class_type} not found in global scope.")

        ClassObj = globals()[class_type]
        instance = ClassObj(**cfg)

        return instance

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None

        return data


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    if isinstance(data, int):
        return torch.LongTensor([data])
    if isinstance(data, float):
        return torch.FloatTensor([data])
    raise TypeError(f"type {type(data)} cannot be converted to tensor.")


class Flip:
    """Flip the input images with a probability."""

    _directions = ["horizontal", "vertical"]

    def __init__(
        self,
        flip_ratio=0.5,
        direction="horizontal",
        flip_label_map=None,
        left_kp=None,
        right_kp=None,
    ):
        if direction not in self._directions:
            raise ValueError(
                f"Direction {direction} is not supported. "
                f"Currently support ones are {self._directions}"
            )
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.flip_label_map = flip_label_map
        self.left_kp = left_kp
        self.right_kp = right_kp

    def _flip_imgs(self, imgs, modality):
        _ = [mmcv.imflip_(img, self.direction) for img in imgs]
        lt = len(imgs)
        if modality == "Flow":
            # The 1st frame of each 2 frames is flow-x
            for i in range(0, lt, 2):
                imgs[i] = mmcv.iminvert(imgs[i])
        return imgs

    def _flip_kps(self, kps, kpscores, img_width):
        kp_x = kps[..., 0]
        kp_x[kp_x != 0] = img_width - kp_x[kp_x != 0]
        new_order = list(range(kps.shape[2]))
        if self.left_kp is not None and self.right_kp is not None:
            for left, right in zip(self.left_kp, self.right_kp):
                new_order[left] = right
                new_order[right] = left
        kps = kps[:, :, new_order]
        if kpscores is not None:
            kpscores = kpscores[:, :, new_order]
        return kps, kpscores

    @staticmethod
    def _box_flip(box, img_width):
        """Flip the bounding boxes given the width of the image.

        Args:
            box (np.ndarray): The bounding boxes.
            img_width (int): The img width.
        """
        box_ = box.copy()
        box_[..., 0::4] = img_width - box[..., 2::4]
        box_[..., 2::4] = img_width - box[..., 0::4]
        return box_

    def __call__(self, results):
        """Performs the Flip augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if "keypoint" in results:
            assert self.direction == "horizontal", (
                "Only horizontal flips are" "supported for human keypoints"
            )

        modality = results["modality"]
        if modality == "Flow":
            assert self.direction == "horizontal"

        flip = np.random.rand() < self.flip_ratio

        results["flip"] = flip
        results["flip_direction"] = self.direction
        img_width = results["img_shape"][1]

        if self.flip_label_map is not None and flip:
            results["label"] = self.flip_label_map.get(results["label"], results["label"])
        if flip:
            if "imgs" in results:
                results["imgs"] = self._flip_imgs(results["imgs"], modality)
            if "keypoint" in results:
                kp = results["keypoint"]
                kpscore = results.get("keypoint_score", None)
                kp, kpscore = self._flip_kps(kp, kpscore, img_width)
                results["keypoint"] = kp
                if "keypoint_score" in results:
                    results["keypoint_score"] = kpscore
            if "label_flip" in results:
                assert results["input_type"] != "hand"
                for k_, v_ in results["label_flip"].items():
                    results[k_] = v_
                del results["label_flip"]
            if "gt_bboxes" in results:
                assert self.direction == "horizontal"
                width = results["img_shape"][1]
                results["gt_bboxes"] = self._box_flip(results["gt_bboxes"], width)

        return results


class Normalize:
    """Normalize images with the given mean and std value.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs" and "img_norm_cfg". If modality is 'Flow', additional
    keys "scale_factor" is required

    Args:
        mean (Sequence[float]): Mean values of different channels.
        std (Sequence[float]): Std values of different channels.
        to_bgr (bool): Whether to convert channels from RGB to BGR.
            Default: False.
        adjust_magnitude (bool): Indicate whether to adjust the flow magnitude
            on 'scale_factor' when modality is 'Flow'. Default: False.
    """

    def __init__(self, mean, std, to_bgr=False, adjust_magnitude=False):
        if not isinstance(mean, Sequence):
            raise TypeError(f"Mean must be list, tuple or np.ndarray, but got {type(mean)}")

        if not isinstance(std, Sequence):
            raise TypeError(f"Std must be list, tuple or np.ndarray, but got {type(std)}")

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_bgr = to_bgr
        self.adjust_magnitude = adjust_magnitude

    def __call__(self, results):
        modality = results["modality"]

        if modality == "RGB" or "RGB" in modality:
            n = len(results["imgs"])
            h, w, c = results["imgs"][0].shape
            imgs = np.empty((n, h, w, c), dtype=np.float32)
            for i, img in enumerate(results["imgs"]):
                imgs[i] = img

            for img in imgs:
                mmcv.imnormalize_(img, self.mean, self.std, self.to_bgr)  # NOTE: this is to rgb

            results["imgs"] = imgs
            results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_bgr=self.to_bgr)
            return results

        raise NotImplementedError


class ToTensor:
    """Convert some values in results dict to `torch.Tensor` type in data
    loader pipeline.

    Args:
        keys (Sequence[str]): Required keys to be converted.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Performs the ToTensor formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        for key in results.keys():
            if (key in self.keys) or (key.endswith("task")):
                results[key] = to_tensor(results[key])

        return results


class Collect:
    """Collect data from the loader relevant to the specific task."""

    def __init__(
        self,
        keys,
        meta_keys=(
            "filename",
            "label",
            "original_shape",
            "img_shape",
            "pad_shape",
            "flip_direction",
            "img_norm_cfg",
        ),
        meta_name="img_metas",
        nested=False,
    ):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name
        self.nested = nested

    def __call__(self, results):
        """Performs the Collect formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        data = {}
        for key in results.keys():
            if (key in self.keys) or (key.endswith("task")):
                data[key] = results[key]

        if len(self.meta_keys) != 0:
            raise
        if self.nested:
            for k in data:
                data[k] = [data[k]]

        return data


class FormatShape(object):
    """Format final imgs shape to the given input_format.

    Required keys are "imgs", "num_clips" and "clip_len", added or modified
    keys are "imgs" and "input_shape".

    Args:
        input_format (str): Define the final imgs format.
    """

    def __init__(self, input_format):
        self.input_format = input_format
        if self.input_format not in ["NCTHW", "NCHW", "NCTHW_Heatmap"]:
            raise ValueError(f"The input format {self.input_format} is invalid.")

    def __call__(self, results):
        """Performs the FormatShape formating.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * L
        if self.input_format == "NCTHW":
            if "imgs" in results:
                imgs = results["imgs"]
                num_clips = results["num_clips"]
                clip_len = results["RGB"]["clip_len"]
                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x L x H x W x C
                imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
                # N_crops x N_clips x C x L x H x W
                imgs = imgs.reshape((-1,) + imgs.shape[2:])
                # M' x C x L x H x W
                # M' = N_crops x N_clips
                results["imgs"] = imgs
                results["input_shape"] = imgs.shape

            if "heatmap_imgs" in results:
                imgs = results["heatmap_imgs"]
                num_clips = results["num_clips"]
                clip_len = results["Pose"]["clip_len"]

                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x L x C x H x W
                imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
                # N_crops x N_clips x C x L x H x W
                imgs = imgs.reshape((-1,) + imgs.shape[2:])
                # M' x C x L x H x W
                # M' = N_crops x N_clips
                results["heatmap_imgs"] = imgs
                results["heatmap_input_shape"] = imgs.shape

        elif self.input_format == "NCHW":
            if "imgs" in results:
                imgs = results["imgs"]
                imgs = np.transpose(imgs, (0, 3, 1, 2))
                # M x C x H x W
                results["imgs"] = imgs
                results["input_shape"] = imgs.shape
        elif self.input_format == "NCTHW_Heatmap":
            if "imgs" in results:
                num_clips = results["num_clips"]
                clip_len = results["Pose"]["clip_len"]
                imgs = results["imgs"]

                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x L x C x H x W
                imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
                # N_crops x N_clips x C x L x H x W
                imgs = imgs.reshape((-1,) + imgs.shape[2:])
                # M' x C x L x H x W
                # M' = N_crops x N_clips
                results["imgs"] = imgs
                results["input_shape"] = imgs.shape

        return results


class Resize:
    """Resize images to a specific size.

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "img_shape", "keep_ratio",
    "scale_factor", "resize_size".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
    """

    def __init__(self, scale, keep_ratio=True, interpolation="bilinear"):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f"Invalid scale {scale}, must be positive.")
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(f"Scale must be float or tuple of int, but got {type(scale)}")
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation

    def _resize_imgs(self, imgs, new_w, new_h):
        return [
            mmcv.imresize(img, (new_w, new_h), interpolation=self.interpolation) for img in imgs
        ]

    @staticmethod
    def _resize_kps(kps, scale_factor):
        return kps * scale_factor

    @staticmethod
    def _box_resize(box, scale_factor):
        """Rescale the bounding boxes according to the scale_factor.

        Args:
            box (np.ndarray): The bounding boxes.
            scale_factor (np.ndarray): The scale factor used for rescaling.
        """
        assert len(scale_factor) == 2
        scale_factor = np.concatenate([scale_factor, scale_factor])
        return box * scale_factor

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if "scale_factor" not in results:
            results["scale_factor"] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results["img_shape"]

        if self.keep_ratio:
            new_w, new_h = mmcv.rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h], dtype=np.float32)

        results["img_shape"] = (new_h, new_w)
        results["keep_ratio"] = self.keep_ratio
        results["scale_factor"] = results["scale_factor"] * self.scale_factor

        if "imgs" in results:
            results["imgs"] = self._resize_imgs(results["imgs"], new_w, new_h)
        if "keypoint" in results:
            results["keypoint"] = self._resize_kps(results["keypoint"], self.scale_factor)
        if "gt_bboxes" in results:
            results["gt_bboxes"] = self._box_resize(results["gt_bboxes"], self.scale_factor)

        return results
