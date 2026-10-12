import os
import subprocess
import pandas as pd

INDEX_PATH = "pilot_grasp_videos/event_index.csv"
OUTPUT_PATH = "pilot_grasp_annotations.csv"
VIDEO_DIR = "pilot_grasp_videos"

index = pd.read_csv(INDEX_PATH)

# Resume previous work if available
if os.path.exists(OUTPUT_PATH):
    annotations = pd.read_csv(OUTPUT_PATH)
else:
    annotations = index.copy()

    annotations["age_suitable"] = ""
    annotations["grasp_label"] = ""
    annotations["confidence"] = ""
    annotations["thumb_visible"] = ""
    annotations["index_visible"] = ""
    annotations["object_visible"] = ""
    annotations["notes"] = ""

def ask_choice(prompt, valid):
    while True:
        ans = input(prompt).strip().lower()

        if ans in valid:
            return ans

        print("Choose one of:", ", ".join(valid))

for i, row in annotations.iterrows():

    # Skip already annotated events
    existing = str(row.get("grasp_label", "")).strip()

    if existing in ["P", "W", "O", "U"]:
        continue

    event = int(row["event_number"])
    video = os.path.join(
        VIDEO_DIR,
        row["video_file"]
    )

    print("\n" + "=" * 70)
    print(f"EVENT {event:02d}")
    print("=" * 70)

    print("Source:", row["video_id"])
    print("Clip:", row["clip"])
    print("PID:", row["pid"])
    print("Hand:", row["hand"])
    print(
        "Frames:",
        row["start_frame"],
        "-",
        row["end_frame"]
    )
    print("Age tier:", row["age_tier"])

    # Open video in default macOS player
    subprocess.run(["open", video])

    print("""
GRASP LABELS
P = Precision / pincer-like grasp
W = Whole-hand / palmar grasp
O = Other grasp
U = Unclear / cannot reliably classify

CONFIDENCE
3 = Clear
2 = Probable
1 = Uncertain
""")

    age = ask_choice(
        "Age suitable? [y/n/u]: ",
        ["y", "n", "u"]
    )

    label = ask_choice(
        "Grasp [p/w/o/u]: ",
        ["p", "w", "o", "u"]
    ).upper()

    confidence = ask_choice(
        "Confidence [1/2/3]: ",
        ["1", "2", "3"]
    )

    thumb = ask_choice(
        "Thumb clearly visible? [y/n]: ",
        ["y", "n"]
    )

    index_finger = ask_choice(
        "Index finger clearly visible? [y/n]: ",
        ["y", "n"]
    )

    obj = ask_choice(
        "Object clearly visible? [y/n]: ",
        ["y", "n"]
    )

    notes = input(
        "Short notes (press Enter if none): "
    ).strip()

    annotations.at[i, "age_suitable"] = age
    annotations.at[i, "grasp_label"] = label
    annotations.at[i, "confidence"] = int(confidence)
    annotations.at[i, "thumb_visible"] = thumb
    annotations.at[i, "index_visible"] = index_finger
    annotations.at[i, "object_visible"] = obj
    annotations.at[i, "notes"] = notes

    # Save after every event
    annotations.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("✅ Saved")

print("\nAnnotation complete.")

annotations.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Saved:", OUTPUT_PATH)
