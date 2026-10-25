import math


# --------------------------------------------------
# Which objects is each hand using?
#
# An object being visible does not mean it is being used. For each
# hand, the distance from its fingertips to each object box is
# measured in palm lengths (scale-free), and combined with the
# interaction model:
#
#   holding   fingertip inside / at the box AND the hand model
#             says the hand is interacting
#   touching  fingertip inside / at the box, but no interaction
#   near      within NEAR palm lengths
#
# Everything here is an estimate from 2D boxes and landmarks;
# a toy behind the hand looks the same as a toy in the hand.
# --------------------------------------------------

FINGERTIPS = [4, 8, 12, 16, 20]
WRIST = 0
MIDDLE_MCP = 9

# Palm lengths
CONTACT = 0.35
NEAR = 1.5


def _point_box_distance(x, y, box):

    x1, y1, x2, y2 = box

    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)

    return math.hypot(dx, dy)


def associate(hand_landmarks, interacting, objects, video_size):

    # hand_landmarks: 21 x [x, y, z] in video pixels (latest frame)
    # objects: [{label, confidence, box: [x1, y1, x2, y2] normalised}]
    # video_size: (width, height) of the video the landmarks came from

    width, height = video_size

    palm = math.hypot(
        hand_landmarks[WRIST][0] - hand_landmarks[MIDDLE_MCP][0],
        hand_landmarks[WRIST][1] - hand_landmarks[MIDDLE_MCP][1]
    )

    palm = max(palm, 1e-6)

    related = []

    for obj in objects:

        box = (
            obj["box"][0] * width,
            obj["box"][1] * height,
            obj["box"][2] * width,
            obj["box"][3] * height
        )

        distance = min(
            _point_box_distance(hand_landmarks[i][0], hand_landmarks[i][1], box)
            for i in FINGERTIPS
        ) / palm

        if distance <= CONTACT:
            relation = "holding" if interacting else "touching"
        elif distance <= NEAR:
            relation = "near"
        else:
            continue

        related.append({
            "label": obj["label"],
            "relation": relation,
            "confidence": obj["confidence"],
            "distance_palms": round(distance, 2)
        })

    order = {"holding": 0, "touching": 1, "near": 2}
    related.sort(key=lambda r: (order[r["relation"]], r["distance_palms"], -r["confidence"]))

    holding = [r for r in related if r["relation"] == "holding"]

    return {
        "object_in_hand": holding[0]["label"] if holding else None,
        "objects_near": related
    }
