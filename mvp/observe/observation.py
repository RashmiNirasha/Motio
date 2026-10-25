from collections import Counter, defaultdict

from observe.safety import object_notes


# --------------------------------------------------
# Turn the recent per-tick history of a session into the
# compact, structured observation that is sent to the LLM.
#
# Only derived facts go to the LLM: no images, no landmarks.
# Each fact keeps its uncertainty (probabilities, "inferred"),
# and a fixed reliability section says how far the vision
# models can be trusted, so the LLM can hedge.
# --------------------------------------------------

WINDOW_S = 60

# Ticks further apart than this (paused video, hand away) don't count as time.
MAX_TICK_S = 1.0

RELIABILITY = {
    "hand_interaction_model": (
        "Estimates from hand shape only whether a hand is handling an object. "
        "In testing it caught about 3 in 4 real interactions, and about half of "
        "its 'interacting' calls were wrong. It cannot tell grasp, hold or "
        "release apart."
    ),
    "phases": (
        "idle / moving / engaging / holding / manipulating / releasing are "
        "inferred from the interaction probability and hand speed, not recognised."
    ),
    "object_detector": (
        "A general-purpose detector; labels can be wrong (e.g. a toy drum as "
        "'cup', wheels as 'ball'). Confidence below 0.4 is uncertain."
    ),
    "object_in_hand": (
        "Based on fingertip distance to the object's box in a 2D image; an object "
        "just behind the hand looks the same as one being held."
    ),
}


def _tick_durations(ticks):

    durations = []

    for current, following in zip(ticks, ticks[1:] + [None]):

        if following is None:
            durations.append(0.25)
        else:
            durations.append(min(max(following["t"] - current["t"], 0.0), MAX_TICK_S))

    return durations


def build_observation(ticks, age_months, window_s=WINDOW_S):

    observation = {
        "child_age_months": age_months,
        "window_seconds": window_s,
        "current": {"hands": [], "objects_visible": []},
        "recent": {},
        "safety_notes": [],
        "reliability": RELIABILITY
    }

    if not ticks:
        observation["recent"] = {"note": "No observation yet (camera not started or no data)."}
        return observation

    now = ticks[-1]["t"]
    recent = [tick for tick in ticks if now - tick["t"] <= window_s]
    durations = _tick_durations(recent)

    # ---- current ----

    latest = recent[-1]

    for hand in latest["hands"]:

        if not hand.get("visible", True):
            continue

        observation["current"]["hands"].append({
            "hand": hand["handedness"],
            "phase": hand["phase"],
            "phase_basis": hand["phase_basis"],
            "interaction_probability": round(hand["p_smooth"], 2),
            "interacting": hand["interacting"],
            "object_in_hand": hand.get("object_in_hand"),
            "objects_near": [
                f"{o['label']} ({o['relation']}, confidence {o['confidence']:.2f})"
                for o in hand.get("objects_near", [])
            ]
        })

    visible = defaultdict(list)

    for obj in latest.get("objects", []):
        visible[obj["label"]].append(obj["confidence"])

    observation["current"]["objects_visible"] = [
        {"label": label, "count": len(c), "max_confidence": round(max(c), 2)}
        for label, c in sorted(visible.items(), key=lambda kv: -max(kv[1]))
    ]

    # ---- recent window ----

    hands_visible_s = 0.0
    interacting_s = 0.0
    handled = Counter()
    seen_s = Counter()
    seen_conf = defaultdict(float)

    for tick, dt in zip(recent, durations):

        visible_hands = [h for h in tick["hands"] if h.get("visible", True)]

        if visible_hands:
            hands_visible_s += dt

        if any(h["interacting"] for h in visible_hands):
            interacting_s += dt

        for h in visible_hands:
            if h.get("object_in_hand"):
                handled[h["object_in_hand"]] += dt

        for label in {o["label"] for o in tick.get("objects", [])}:
            seen_s[label] += dt

        for o in tick.get("objects", []):
            seen_conf[o["label"]] = max(seen_conf[o["label"]], o["confidence"])

    # Interaction episodes per hand, with the object most often in hand.
    episodes = []
    open_episodes = {}

    for tick, dt in zip(recent, durations):

        present = set()

        for h in tick["hands"]:

            slot = h["slot"]
            present.add(slot)

            if h["interacting"]:

                episode = open_episodes.setdefault(
                    slot,
                    {"hand": h["handedness"], "start": tick["t"], "seconds": 0.0, "objects": Counter()}
                )

                episode["seconds"] += dt

                if h.get("object_in_hand"):
                    episode["objects"][h["object_in_hand"]] += 1

            elif slot in open_episodes:
                episodes.append(open_episodes.pop(slot))

        for slot in list(open_episodes):
            if slot not in present:
                episodes.append(open_episodes.pop(slot))

    ongoing = list(open_episodes.values())

    def describe(episode, is_ongoing):

        top = episode["objects"].most_common(1)

        return {
            "hand": episode["hand"],
            "seconds": round(episode["seconds"], 1),
            "object": top[0][0] if top else None,
            "ongoing": is_ongoing,
            "seconds_ago": round(now - episode["start"], 1)
        }

    observation["recent"] = {
        "seconds_observed": round(sum(durations), 1),
        "seconds_hands_visible": round(hands_visible_s, 1),
        "seconds_interacting": round(interacting_s, 1),
        "interaction_episodes": (
            [describe(e, False) for e in episodes] + [describe(e, True) for e in ongoing]
        )[-8:],
        "objects_handled_seconds": {k: round(v, 1) for k, v in handled.most_common()},
        "objects_seen": [
            {"label": label, "seconds_visible": round(s, 1), "max_confidence": round(seen_conf[label], 2)}
            for label, s in seen_s.most_common(10)
        ]
    }

    observation["safety_notes"] = object_notes(
        [o["label"] for o in observation["recent"]["objects_seen"]],
        age_months
    )

    return observation
