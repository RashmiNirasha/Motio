import json
import os
import time
from collections import deque

from observe.association import associate
from observe.hand_state import HandReading, HandStateMachine, ObserveConfig
from observe.observation import build_observation


# Objects older than this are not linked to hands.
OBJECT_FRESH_S = 2.5

# History kept for the observation summary.
HISTORY_S = 180


# --------------------------------------------------
# One observation session (one caregiver, one camera).
#
# Wraps the hand state machine and can record each tick as
# one JSON line. Logs contain only numbers (probabilities,
# wrist positions, derived states), never images or
# landmarks, and can be replayed offline with replay.py.
# --------------------------------------------------

class ObservationSession:

    def __init__(self, session_id, config=None, log_dir=None):

        self.session_id = session_id
        self.config = config or ObserveConfig()
        self.machine = HandStateMachine(self.config)
        self.log_dir = log_dir

        self.created = time.time()
        self.last_result = None
        self.last_objects = None
        self.log_path = None
        self._log = None

        # Set by the caregiver in the UI
        self.child_age_months = None

        # Recent ticks (hands with object links, objects) for the summary
        self.ticks = deque()

        # Conversation with the assistant (list of API messages)
        self.chat = []

    # ------------------------------------------------

    def update(self, t, readings, hand_frames=None, video_size=None):

        # hand_frames: {slot: 21 x [x, y, z] latest landmarks in video pixels}

        states, events = self.machine.update(t, readings)

        hands = [state.to_dict() for state in states]

        objects = []

        if self.last_objects and t - self.last_objects["t"] <= OBJECT_FRESH_S:
            objects = self.last_objects["objects"]

        for hand in hands:

            frame = (hand_frames or {}).get(hand["slot"])

            if frame is not None and video_size and objects and hand["visible"]:
                hand.update(associate(frame, hand["interacting"], objects, video_size))
            else:
                hand.update({"object_in_hand": None, "objects_near": []})

        self.ticks.append({"t": t, "hands": hands, "objects": objects})

        while self.ticks and t - self.ticks[0]["t"] > HISTORY_S:
            self.ticks.popleft()

        result = {
            "t": round(t, 3),
            "hands": hands,
            "events": events
        }

        if self._log:

            self._log.write(
                json.dumps({
                    "t": round(t, 3),
                    "readings": [
                        {
                            "slot": r.slot,
                            "p": round(r.p, 5),
                            "wrist": [round(r.wrist[0], 2), round(r.wrist[1], 2)],
                            "palm": round(r.palm, 2),
                            "handedness": r.handedness
                        }
                        for r in readings
                    ],
                    "hands": result["hands"],
                    "events": events
                })
                + "\n"
            )

            self._log.flush()

        self.last_result = result

        return result

    def reset(self):

        self.machine.reset()
        self.last_result = None
        self.last_objects = None
        self.ticks.clear()
        self.chat = []

    def observation(self):

        return build_observation(list(self.ticks), self.child_age_months)

    # ------------------------------------------------
    # Recording
    # ------------------------------------------------

    @property
    def recording(self):

        return self._log is not None

    def start_recording(self):

        if self._log or not self.log_dir:
            return self.log_path

        os.makedirs(self.log_dir, exist_ok=True)

        self.log_path = os.path.join(
            self.log_dir,
            time.strftime("%Y%m%d-%H%M%S") + f"_{self.session_id[:8]}.jsonl"
        )

        self._log = open(self.log_path, "w")

        # First line describes the settings the log was made with,
        # so replay can use the same ones.
        self._log.write(
            json.dumps({"config": self.config.__dict__}) + "\n"
        )

        return self.log_path

    def stop_recording(self):

        if self._log:
            self._log.close()
            self._log = None


class SessionStore:

    # Single-user MVP: keep a few sessions, drop the oldest.

    def __init__(self, config=None, log_dir=None, max_sessions=20):

        self.config = config
        self.log_dir = log_dir
        self.max_sessions = max_sessions
        self.sessions = {}

    def get(self, session_id):

        if session_id not in self.sessions:

            if len(self.sessions) >= self.max_sessions:

                oldest = min(
                    self.sessions.values(),
                    key=lambda s: s.created
                )

                oldest.stop_recording()
                del self.sessions[oldest.session_id]

            self.sessions[session_id] = ObservationSession(
                session_id,
                config=self.config,
                log_dir=self.log_dir
            )

        return self.sessions[session_id]


__all__ = [
    "HandReading",
    "ObservationSession",
    "ObserveConfig",
    "SessionStore"
]
