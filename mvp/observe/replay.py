import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from observe.hand_state import HandReading, HandStateMachine, ObserveConfig


# --------------------------------------------------
# Replay a recorded observation session (from the
# "Record" button on /app/observe.html) with no camera.
#
#   python mvp/observe/replay.py mvp/observe_logs/<file>.jsonl
#   python mvp/observe/replay.py <file> --moving-speed 1.5
#
# Re-runs the hand state machine on the logged model
# probabilities and wrist positions, prints the timeline,
# and, when using the logged settings, checks that the replay
# reproduces the logged phases exactly.
# --------------------------------------------------

def load(path):

    config = {}
    ticks = []

    with open(path) as f:

        for line in f:

            row = json.loads(line)

            if "config" in row:
                config = row["config"]
            else:
                ticks.append(row)

    return config, ticks


def replay(config, ticks):

    machine = HandStateMachine(ObserveConfig(**config))

    timeline = []
    mismatches = 0

    for tick in ticks:

        readings = [
            HandReading(
                slot=r["slot"],
                p=r["p"],
                wrist=tuple(r["wrist"]),
                palm=r["palm"],
                handedness=r["handedness"]
            )
            for r in tick["readings"]
        ]

        states, events = machine.update(tick["t"], readings)

        phases = {s.slot: s.phase for s in states}
        logged = {h["slot"]: h["phase"] for h in tick["hands"]}

        if phases != logged:
            mismatches += 1

        timeline.append((tick["t"], phases, events))

    return timeline, mismatches


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--moving-speed", type=float)
    parser.add_argument("--manipulating-speed", type=float)
    args = parser.parse_args()

    config, ticks = load(args.log)

    changed = {}

    for name in ["threshold", "moving_speed", "manipulating_speed"]:

        value = getattr(args, name)

        if value is not None:
            config[name] = value
            changed[name] = value

    timeline, mismatches = replay(config, ticks)

    if not ticks:
        sys.exit("No ticks in this log.")

    duration = ticks[-1]["t"] - ticks[0]["t"]

    print(f"{len(ticks)} ticks over {duration:.1f} s")
    print("settings:", json.dumps(config))

    time_in_phase = {}

    for (t, phases, events), nxt in zip(timeline, timeline[1:]):

        for slot, phase in phases.items():
            time_in_phase.setdefault(slot, {})
            time_in_phase[slot][phase] = (
                time_in_phase[slot].get(phase, 0.0) + (nxt[0] - t)
            )

        for event in events:

            detail = ""

            if event["type"] == "interaction_ended":
                detail = f"  lasted {event['duration_s']} s, mean p {event['mean_p']}"

            print(f"  t={event['t']:8.2f}  hand {event['slot']}  {event['type']}{detail}")

    print("\nTime in each inferred phase (seconds):")

    for slot, phases in sorted(time_in_phase.items()):
        print(f"  hand {slot}: " + ", ".join(f"{p} {s:.1f}" for p, s in sorted(phases.items())))

    if changed:
        print(f"\nReplayed with changed settings {changed}; phases are not compared with the log.")
    elif mismatches:
        print(f"\nWARNING: {mismatches} ticks differ from the logged phases.")
    else:
        print("\nReplay matches the logged phases exactly.")


if __name__ == "__main__":
    main()
