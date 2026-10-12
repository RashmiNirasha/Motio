import pandas as pd

events = pd.read_csv("verified_child_grasp_events.csv")
clips = pd.read_csv("data/ChildPlay-gaze/clips.csv")

events = events.merge(
    clips[["clip", "video_id", "fps", "resolution"]],
    on="clip",
    how="left"
)

tier_a = [
    "LHT2zVYvObg",
    "w6oRoyTBmU4",
    "zUdMBtK2UVE"
]

tier_b = [
    "iWYy8n6mSwc",
    "DGls-Z4e6OI",
    "8db0Gcv3V10"
]

events["age_tier"] = "exclude"

events.loc[
    events["video_id"].isin(tier_a),
    "age_tier"
] = "A"

events.loc[
    events["video_id"].isin(tier_b),
    "age_tier"
] = "B"

usable = events[
    events["age_tier"].isin(["A", "B"])
].copy()

usable = usable.sort_values(
    ["age_tier", "mean_visibility"],
    ascending=[True, False]
)

# Avoid taking too many nearly identical events from one clip
pilot = (
    usable
    .drop_duplicates(subset=["clip", "pid", "hand"])
    .head(20)
    .copy()
)

pilot["age_suitable"] = ""
pilot["grasp_label"] = ""
pilot["confidence"] = ""
pilot["thumb_visible"] = ""
pilot["index_visible"] = ""
pilot["object_visible"] = ""
pilot["notes"] = ""

pilot.to_csv(
    "infant_pilot_20.csv",
    index=False
)

print("\nSELECTED PILOT EVENTS\n")

print(
    pilot[
        [
            "age_tier",
            "video_id",
            "clip",
            "pid",
            "hand",
            "start_frame",
            "end_frame",
            "mean_visibility"
        ]
    ].to_string(index=False)
)

print("\nCounts by source:")
print(pilot["video_id"].value_counts())

print("\nSaved: infant_pilot_20.csv")
