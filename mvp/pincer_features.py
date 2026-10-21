import numpy as np


# --------------------------------------------------
# Pincer-grasp features from MediaPipe landmarks
#
# Shared by the FastAPI server (live inference),
# and train_pincer.py (training), so both always
# use exactly the same representation.
#
# Input landmarks:
#   T × 21 × 3 in isotropic pixel units
#   (x * videoWidth, y * videoHeight, z * videoWidth)
# --------------------------------------------------

WRIST = 0
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

FINGERTIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

# 63 relative coordinates
# + 4 thumb-tip → fingertip distances
# + 4 fingertip → wrist distances (finger curl)
# + 3 joint angle cosines
FEATURE_DIM = 74

SEQUENCE_LENGTH = 20

LABELS = ["pincer", "other_grasp", "no_grasp"]


def _distance(landmarks, a, b):

    return np.linalg.norm(
        landmarks[:, a] - landmarks[:, b],
        axis=-1
    )


def _joint_cosine(landmarks, a, b, c):

    # Cosine of the angle at joint b.
    # 1 = fully bent back on itself, -1 = straight.
    u = landmarks[:, a] - landmarks[:, b]
    v = landmarks[:, c] - landmarks[:, b]

    return (
        np.sum(u * v, axis=-1)
        /
        (
            np.linalg.norm(u, axis=-1)
            * np.linalg.norm(v, axis=-1)
            + 1e-6
        )
    )


def _palm_scale(landmarks):

    # Wrist → middle-finger knuckle.
    # Unlike the bounding box, this does not
    # shrink when the fingers close.
    return np.maximum(
        _distance(landmarks, WRIST, MIDDLE_MCP),
        1e-6
    )


def landmarks_to_features(landmarks):

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32
    )

    if landmarks.ndim != 3 or landmarks.shape[1:] != (21, 3):

        raise ValueError(
            "Expected landmarks [T,21,3], "
            f"received {list(landmarks.shape)}"
        )

    scale = _palm_scale(landmarks)

    relative = (
        landmarks - landmarks[:, WRIST:WRIST + 1]
    ) / scale[:, None, None]

    thumb_to_tips = np.stack(
        [
            _distance(landmarks, THUMB_TIP, tip)
            for tip in FINGERTIPS
        ],
        axis=1
    ) / scale[:, None]

    tips_to_wrist = np.stack(
        [
            _distance(landmarks, tip, WRIST)
            for tip in FINGERTIPS
        ],
        axis=1
    ) / scale[:, None]

    angles = np.stack(
        [
            _joint_cosine(landmarks, INDEX_MCP, INDEX_PIP, INDEX_DIP),
            _joint_cosine(landmarks, INDEX_PIP, INDEX_DIP, INDEX_TIP),
            _joint_cosine(landmarks, 2, THUMB_IP, THUMB_TIP)
        ],
        axis=1
    )

    features = np.concatenate(
        [
            relative.reshape(len(landmarks), 63),
            thumb_to_tips,
            tips_to_wrist,
            angles
        ],
        axis=1
    )

    return features.astype(np.float32)


# --------------------------------------------------
# Rule-based fallback
#
# Used until a trained pincer model exists.
# Thresholds are rough starting values (palm-size units)
# and should be checked against real recordings.
# --------------------------------------------------

PINCH_GAP = 0.35
INDEX_CURLED = 1.0


def _sigmoid(x):

    return 1.0 / (1.0 + np.exp(-x))


def heuristic_pincer_probability(landmarks, recent_frames=5):

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32
    )[-recent_frames:]

    scale = _palm_scale(landmarks)

    thumb_index_gap = (
        _distance(landmarks, THUMB_TIP, INDEX_TIP) / scale
    )

    thumb_middle_gap = (
        _distance(landmarks, THUMB_TIP, MIDDLE_TIP) / scale
    )

    index_reach = (
        _distance(landmarks, INDEX_TIP, WRIST) / scale
    )

    # Thumb and index tips touching.
    pinching = _sigmoid(
        (PINCH_GAP - thumb_index_gap) / 0.06
    )

    # Index finger not rolled into a fist
    # (fist / palmar grasp also brings the thumb near the index).
    index_not_curled = _sigmoid(
        (index_reach - INDEX_CURLED) / 0.1
    )

    # Thumb meets the index more than the middle finger
    # (separates pincer from whole-hand pinches).
    index_led = _sigmoid(
        (thumb_middle_gap - thumb_index_gap - 0.1) / 0.06
    )

    return float(
        np.mean(pinching * index_not_curled * index_led)
    )
