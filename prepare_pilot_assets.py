import pandas as pd

pilot = pd.read_csv("infant_pilot_20.csv")
clips = pd.read_csv("data/ChildPlay-gaze/clips.csv")
videos = pd.read_csv("data/ChildPlay-gaze/videos.csv")

# Unique source videos
source_ids = pilot["video_id"].drop_duplicates().tolist()

sources = videos[
    videos["video_id"].isin(source_ids)
].copy()

sources["youtube_url"] = (
    "https://www.youtube.com/watch?v="
    + sources["video_id"]
)

sources.to_csv(
    "pilot_source_videos.csv",
    index=False
)

sources["youtube_url"].to_csv(
    "pilot_source_urls.txt",
    index=False,
    header=False
)

# Only clips required by the 20-event pilot
needed_clips = clips[
    clips["clip"].isin(pilot["clip"])
].copy()

needed_clips.to_csv(
    "data/ChildPlay-gaze/infant_pilot_clips.csv",
    index=False
)

print("\nSOURCE VIDEOS TO DOWNLOAD")
print("=" * 70)

print(
    sources[
        [
            "video_id",
            "fps",
            "resolution",
            "width",
            "height",
            "frame_count"
        ]
    ].to_string(index=False)
)

print("\nNumber of source videos:", len(sources))
print("Number of ChildPlay clips:", len(needed_clips))

print("\nSaved:")
print("pilot_source_videos.csv")
print("pilot_source_urls.txt")
print("data/ChildPlay-gaze/infant_pilot_clips.csv")
