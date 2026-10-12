import os
import subprocess
import pandas as pd

INPUT = "pilot_grasp_annotations.csv"
OUTPUT = "pilot_reannotation.csv"
VIDEO_DIR = "pilot_grasp_videos"

df = pd.read_csv(INPUT)

low = df[df["confidence"] == 1].copy()

if os.path.exists(OUTPUT):
    old = pd.read_csv(OUTPUT)

    for col in [
        "stable_hold",
        "stable_grasp_label",
        "new_confidence",
        "uncertainty_reason",
        "reannotation_notes"
    ]:
        if col in old.columns:
            low = low.drop(columns=[col], errors="ignore").merge(
                old[["event_number", col]],
                on="event_number",
                how="left"
            )

for col in [
    "stable_hold",
    "stable_grasp_label",
    "new_confidence",
    "uncertainty_reason",
    "reannotation_notes"
]:
    if col not in low.columns:
        low[col] = ""

def ask(prompt, valid):
    while True:
        ans = input(prompt).strip().lower()
        if ans in valid:
            return ans
        print("Choose:", ", ".join(valid))

for i, row in low.iterrows():

    if str(row["stable_grasp_label"]) in ["TIO", "PAL", "IND"]:
        continue

    event = int(row["event_number"])

    video = os.path.join(
        VIDEO_DIR,
        f"event_{event:02d}.mp4"
    )

    print("\n" + "=" * 70)
    print(f"EVENT {event:02d}")
    print("=" * 70)

    print("Original label:", row["grasp_label"])
    print("Hand:", row["hand"])

    subprocess.run(["open", video])

    print("""
Focus ONLY on the stable holding phase.

TIO = thumb-index opposition clearly controls object
PAL = object mainly enclosed/stabilized by palm + multiple fingers
IND = cannot reliably determine
""")

    stable = ask(
        "Is there a stable holding phase? [y/n]: ",
        ["y", "n"]
    )

    if stable == "n":
        label = "IND"
    else:
        label = ask(
            "Stable grasp [tio/pal/ind]: ",
            ["tio", "pal", "ind"]
        ).upper()

    confidence = ask(
        "Confidence [1/2/3]: ",
        ["1", "2", "3"]
    )

    print("""
Uncertainty reason:
1 = grasp changes during event
2 = fingers overlap / configuration unclear
3 = multiple hands involved
4 = object geometry makes grip ambiguous
5 = hand/object partially occluded
6 = no stable grasp
7 = other
8 = none
""")

    reason = ask(
        "Reason [1-8]: ",
        ["1","2","3","4","5","6","7","8"]
    )

    notes = input("Notes: ").strip()

    low.at[i, "stable_hold"] = stable
    low.at[i, "stable_grasp_label"] = label
    low.at[i, "new_confidence"] = int(confidence)
    low.at[i, "uncertainty_reason"] = reason
    low.at[i, "reannotation_notes"] = notes

    low.to_csv(OUTPUT, index=False)

    print("✅ Saved")

print("\nDone.")
print("Saved:", OUTPUT)
