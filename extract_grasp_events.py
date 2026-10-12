import pandas as pd

# Load the grasp rows we created earlier
df = pd.read_csv("grasp_annotations.csv")

# Sort so consecutive frames are together
df = df.sort_values(
    ["split", "source_file", "pid", "hand", "frame"]
).reset_index(drop=True)

# A new event starts when:
# 1. clip changes
# 2. person changes
# 3. hand changes
# 4. frame is not consecutive
new_event = (
    (df["source_file"] != df["source_file"].shift()) |
    (df["pid"] != df["pid"].shift()) |
    (df["hand"] != df["hand"].shift()) |
    ((df["frame"] - df["frame"].shift()) > 1)
)

df["event_id"] = new_event.cumsum()

events = (
    df.groupby("event_id")
    .agg(
        split=("split", "first"),
        source_file=("source_file", "first"),
        clip=("clip", "first"),
        pid=("pid", "first"),
        hand=("hand", "first"),
        start_frame=("frame", "min"),
        end_frame=("frame", "max"),
        n_frames=("frame", "count"),
        mean_visibility=("h_visible", "mean"),
        min_visibility=("h_visible", "min"),
        max_visibility=("h_visible", "max"),
        oid=("oid", "first"),
    )
    .reset_index()
)

print("\n==============================")
print("GRASP EVENT SUMMARY")
print("==============================")

print("\nTotal grasp events:")
print(len(events))

print("\nEvents by split:")
print(events["split"].value_counts())

print("\nFrames per grasp event:")
print(events["n_frames"].describe())

print("\nMean hand visibility per event:")
print(events["mean_visibility"].describe())

high_vis = events[events["mean_visibility"] >= 0.8]

print("\nEvents with mean visibility >= 0.8:")
print(len(high_vis))

print("\nEvents with >= 5 frames:")
print(len(events[events["n_frames"] >= 5]))

good = events[
    (events["mean_visibility"] >= 0.8) &
    (events["n_frames"] >= 5)
]

print("\nGood candidate events:")
print(len(good))

print("\nExample good events:")
print(
    good[
        [
            "clip",
            "pid",
            "hand",
            "start_frame",
            "end_frame",
            "n_frames",
            "mean_visibility",
            "split"
        ]
    ]
    .head(25)
    .to_string(index=False)
)

events.to_csv("grasp_events.csv", index=False)
good.to_csv("good_grasp_events.csv", index=False)

print("\nSaved:")
print("grasp_events.csv")
print("good_grasp_events.csv")
