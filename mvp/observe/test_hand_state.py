import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from observe.hand_state import HandReading, HandStateMachine, ObserveConfig


# Run from the repository root:
#   python mvp/observe/test_hand_state.py

PALM = 100.0
DT = 0.25


def run(steps, config=None):

    # steps: list of (p, wrist_x) for hand 0, one per 0.25 s tick.
    machine = HandStateMachine(config)
    out = []
    t = 0.0

    for p, x in steps:

        t += DT

        states, events = machine.update(
            t,
            [HandReading(0, p, (x, 100.0), PALM, "Right")]
        )

        out.append((t, states[0] if states else None, events))

    return machine, out


def kinds(out):

    return [e["type"] for _, _, events in out for e in events]


def test_hand_still_and_no_interaction_is_idle():

    _, out = run([(0.1, 50.0)] * 12)

    assert out[-1][1].phase == "idle"
    assert kinds(out) == ["hand_appeared"]


def test_interaction_needs_three_of_four_votes():

    # one stray positive must not confirm interaction
    _, out = run([(0.1, 50.0)] * 4 + [(0.9, 50.0)] + [(0.1, 50.0)] * 6)

    assert "interaction_started" not in kinds(out)

    # sustained positives do
    _, out = run([(0.1, 50.0)] * 4 + [(0.9, 50.0)] * 6)

    assert "interaction_started" in kinds(out)


def test_phase_sequence_engage_hold_release():

    # still hand: idle, then interaction (still), then it ends
    steps = [(0.1, 50.0)] * 4 + [(0.9, 50.0)] * 14 + [(0.1, 50.0)] * 8

    _, out = run(steps)

    phases = [s.phase for _, s, _ in out]

    assert phases[3] == "idle"
    assert "engaging" in phases
    assert "holding" in phases
    assert "releasing" in phases
    assert phases.index("engaging") < phases.index("holding") < phases.index("releasing")
    assert phases[-1] == "idle"

    ended = [e for _, _, ev in out for e in ev if e["type"] == "interaction_ended"]
    assert len(ended) == 1 and ended[0]["duration_s"] > 2


def test_moving_hand_while_interacting_is_manipulating():

    # wrist moves 30 px (0.3 palms) per 0.25 s = 1.2 palms/s
    steps = [(0.9, 50.0 + 30.0 * i) for i in range(20)]

    _, out = run(steps)

    assert out[-1][1].phase == "manipulating"
    assert out[-1][1].speed > 0.6


def test_moving_empty_hand_is_moving_not_interacting():

    steps = [(0.1, 50.0 + 60.0 * i) for i in range(12)]

    _, out = run(steps)

    assert out[-1][1].phase == "moving"
    assert not out[-1][1].interacting


def test_lost_hand_ends_interaction():

    machine, out = run([(0.9, 50.0)] * 12)

    assert out[-1][1].interacting

    # hand disappears: ticks with no readings
    events = []
    t = out[-1][0]

    for _ in range(8):
        t += DT
        _, ev = machine.update(t, [])
        events += ev

    types = [e["type"] for e in events]

    assert "interaction_ended" in types
    assert "hand_lost" in types
    assert machine.tracks == {}

    ended = [e for e in events if e["type"] == "interaction_ended"][0]
    assert ended["reason"] == "hand_lost"


def test_two_hands_are_independent():

    machine = HandStateMachine()
    t = 0.0

    for _ in range(10):

        t += DT

        states, _ = machine.update(
            t,
            [
                HandReading(0, 0.9, (50.0, 100.0), PALM, "Right"),
                HandReading(1, 0.1, (400.0, 100.0), PALM, "Left")
            ]
        )

    by_slot = {s.slot: s for s in states}

    assert by_slot[0].interacting and not by_slot[1].interacting


def test_probability_threshold_is_configurable():

    steps = [(0.5, 50.0)] * 10

    _, out = run(steps, ObserveConfig(threshold=0.66))
    assert "interaction_started" not in kinds(out)

    _, out = run(steps, ObserveConfig(threshold=0.4))
    assert "interaction_started" in kinds(out)


if __name__ == "__main__":

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

    for test in tests:
        test()
        print("ok  ", test.__name__)

    print(f"\n{len(tests)} tests passed")
