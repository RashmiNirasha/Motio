import math
from collections import deque
from dataclasses import dataclass, field


# --------------------------------------------------
# Temporal state for each tracked hand.
#
# Input per hand, a few times per second:
#   - p: probability from the ChildPlay interaction model
#   - wrist position and palm size (pixels)
#
# The interaction model outputs ONE number: "this hand
# appears to be interacting with an object". It cannot tell
# grasp / hold / release apart. The phases below are
# INFERRED from how p and the hand's speed change over time,
# and are always reported with phase_basis = "inferred".
# --------------------------------------------------

PHASE_BASIS = "inferred"

# absent    hand not tracked (not returned)
# idle      hand visible, no interaction, little motion
# moving    hand visible, no interaction, moving
# engaging  interaction just confirmed  (grasp onset, inferred)
# holding   interaction sustained, hand moving little
# manipulating  interaction sustained, hand moving
# releasing interaction just ended
PHASES = ["idle", "moving", "engaging", "holding", "manipulating", "releasing"]


@dataclass
class ObserveConfig:

    # Probability cut-off, same as the deployed model (iphone_threshold.txt)
    threshold: float = 0.66

    # Confirm interaction when enter_votes of the last `window`
    # predictions are positive, end it when exit_votes are negative.
    window: int = 4
    enter_votes: int = 3
    exit_votes: int = 3

    # Smoothing of p and of hand speed (exponential moving average)
    p_alpha: float = 0.5
    speed_alpha: float = 0.5

    # A hand not seen for this long is a lost track
    lost_after_s: float = 1.0

    # Hand speed in palm-lengths per second (scale-free).
    # Starting values, not calibrated: tune from recorded sessions.
    moving_speed: float = 1.0
    manipulating_speed: float = 0.6

    # How long the engaging / releasing phases last
    engage_s: float = 1.0
    release_s: float = 1.0


@dataclass
class HandReading:

    slot: int
    p: float
    wrist: tuple
    palm: float
    handedness: str = "Unknown"


@dataclass
class HandState:

    slot: int
    handedness: str
    visible: bool
    p: float
    p_smooth: float
    interacting: bool
    phase: str
    speed: float
    interaction_s: float
    phase_basis: str = PHASE_BASIS

    def to_dict(self):

        return {
            "slot": self.slot,
            "handedness": self.handedness,
            "visible": self.visible,
            "p": round(self.p, 4),
            "p_smooth": round(self.p_smooth, 4),
            "interacting": self.interacting,
            "phase": self.phase,
            "phase_basis": self.phase_basis,
            "speed_palms_per_s": round(self.speed, 3),
            "interaction_s": round(self.interaction_s, 2)
        }


@dataclass
class _Track:

    slot: int
    handedness: str
    first_t: float
    last_t: float
    window: int
    votes: deque = field(default_factory=deque)
    p: float = 0.0
    p_smooth: float = 0.0
    speed: float = 0.0
    wrist: tuple = None
    palm: float = 1.0
    interacting: bool = False
    enter_t: float = None
    exit_t: float = None
    p_sum: float = 0.0
    p_count: int = 0

    def __post_init__(self):
        self.votes = deque(maxlen=self.window)


class HandStateMachine:

    def __init__(self, config=None):

        self.config = config or ObserveConfig()
        self.tracks = {}

    # ------------------------------------------------
    # One tick: all hands scored at time t (seconds).
    # Returns (states, events).
    # ------------------------------------------------

    def update(self, t, readings):

        events = []
        seen = set()

        for reading in readings:

            seen.add(reading.slot)
            events += self._update_track(t, reading)

        for slot in list(self.tracks):

            track = self.tracks[slot]

            if slot not in seen and t - track.last_t > self.config.lost_after_s:
                events += self._end_track(t, track)

        states = [
            self._state(t, track)
            for track in self.tracks.values()
        ]

        return states, events

    def reset(self):

        self.tracks = {}

    # ------------------------------------------------

    def _update_track(self, t, reading):

        cfg = self.config
        events = []

        track = self.tracks.get(reading.slot)

        if track is None:

            track = _Track(
                slot=reading.slot,
                handedness=reading.handedness,
                first_t=t,
                last_t=t,
                window=cfg.window
            )

            track.p_smooth = reading.p
            self.tracks[reading.slot] = track

            events.append(self._event("hand_appeared", t, track))

        else:

            dt = t - track.last_t

            if dt > 0 and track.wrist is not None:

                palm = max((track.palm + reading.palm) / 2, 1e-6)

                moved = math.hypot(
                    reading.wrist[0] - track.wrist[0],
                    reading.wrist[1] - track.wrist[1]
                )

                instant = min(moved / palm / dt, 20.0)

                track.speed += cfg.speed_alpha * (instant - track.speed)

            track.p_smooth += cfg.p_alpha * (reading.p - track.p_smooth)

        track.handedness = reading.handedness
        track.last_t = t
        track.wrist = reading.wrist
        track.palm = reading.palm
        track.p = reading.p
        track.votes.append(reading.p >= cfg.threshold)

        positives = sum(track.votes)
        negatives = len(track.votes) - positives
        full = len(track.votes) == cfg.window

        if full and not track.interacting and positives >= cfg.enter_votes:

            track.interacting = True
            track.enter_t = t
            track.p_sum = 0.0
            track.p_count = 0

            events.append(self._event("interaction_started", t, track))

        elif full and track.interacting and negatives >= cfg.exit_votes:

            track.interacting = False
            track.exit_t = t

            events.append(
                self._event(
                    "interaction_ended",
                    t,
                    track,
                    duration_s=round(t - track.enter_t, 2),
                    mean_p=round(track.p_sum / max(track.p_count, 1), 3)
                )
            )

        if track.interacting:
            track.p_sum += reading.p
            track.p_count += 1

        return events

    def _end_track(self, t, track):

        events = []

        if track.interacting:

            events.append(
                self._event(
                    "interaction_ended",
                    t,
                    track,
                    duration_s=round(track.last_t - track.enter_t, 2),
                    mean_p=round(track.p_sum / max(track.p_count, 1), 3),
                    reason="hand_lost"
                )
            )

        events.append(self._event("hand_lost", t, track))

        del self.tracks[track.slot]

        return events

    def _state(self, t, track):

        cfg = self.config

        if track.interacting:

            if t - track.enter_t < cfg.engage_s:
                phase = "engaging"
            elif track.speed >= cfg.manipulating_speed:
                phase = "manipulating"
            else:
                phase = "holding"

        elif track.exit_t is not None and t - track.exit_t < cfg.release_s:
            phase = "releasing"

        elif track.speed >= cfg.moving_speed:
            phase = "moving"

        else:
            phase = "idle"

        return HandState(
            slot=track.slot,
            handedness=track.handedness,
            visible=track.last_t == t,
            p=track.p,
            p_smooth=track.p_smooth,
            interacting=track.interacting,
            phase=phase,
            speed=track.speed,
            interaction_s=(t - track.enter_t) if track.interacting else 0.0
        )

    @staticmethod
    def _event(kind, t, track, **extra):

        return {
            "type": kind,
            "t": round(t, 3),
            "slot": track.slot,
            "handedness": track.handedness,
            **extra
        }
