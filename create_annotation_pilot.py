import pandas as pd

events = pd.read_csv(
    "verified_child_grasp_events.csv"
)

clips = pd.read_csv(
    "data/ChildPlay-gaze/clips.csv"
)

events = events.merge(
    clips[
        [
            "clip",
            "video_id",
            "fps",
            "resolution"
        ]
    ],
    on="clip",
    how="left"
)

events = events.sort_values(
    "mean_visibility",
    ascending=False
)

# Prefer diversity
pilot = (
    events
    .drop_duplicates(subset="clip")
    .head(20)
    .copy()
)

# Empty columns for our new annotation
pilot["age_suitable"] = ""
pilot["grasp_label"] = ""
pilot["confidence"] = ""
pilot["object_visible"] = ""
pilot["fingers_visible"] = ""
pilot["notes"] = ""

pilot.to_csv(
    "pilot_20_annotations.csv",
    index=False
)

print(
    pilot[
        [
            "clip",
            "video_id",
            "pid",
            "hand",
            "start_frame",
            "end_frame",
            "mean_visibility"
        ]
    ].to_string(index=False)
)

print("\nSaved: pilot_20_annotations.csv")
