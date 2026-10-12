import pandas as pd

events = pd.read_csv("good_grasp_events.csv")
clips = pd.read_csv("data/ChildPlay-gaze/clips.csv")

# Connect grasp events to their original YouTube video
data = events.merge(
    clips[["clip", "video_id", "fps", "resolution"]],
    on="clip",
    how="left"
)

# Highest visibility first
data = data.sort_values(
    "mean_visibility",
    ascending=False
)

# Choose one strong event from each source video
pilot = (
    data
    .drop_duplicates(subset="video_id")
    .head(10)
    .copy()
)

# YouTube URL
pilot["youtube_url"] = (
    "https://www.youtube.com/watch?v="
    + pilot["video_id"].astype(str)
)

# Save detailed event information
pilot.to_csv(
    "pilot_grasp_events.csv",
    index=False
)

# Create subset of clips.csv for the official extraction script
pilot_clips = clips[
    clips["clip"].isin(pilot["clip"])
]

pilot_clips.to_csv(
    "data/ChildPlay-gaze/pilot_clips.csv",
    index=False
)

# URLs for downloading
pilot["youtube_url"].to_csv(
    "pilot_urls.txt",
    index=False,
    header=False
)

print("\nSelected pilot grasp events:\n")

print(
    pilot[
        [
            "clip",
            "video_id",
            "hand",
            "start_frame",
            "end_frame",
            "n_frames",
            "mean_visibility",
            "fps",
            "resolution"
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print("pilot_grasp_events.csv")
print("pilot_urls.txt")
print("data/ChildPlay-gaze/pilot_clips.csv")
