import pandas as pd

videos = pd.read_csv("data/ChildPlay-gaze/videos.csv")
clips = pd.read_csv("data/ChildPlay-gaze/clips.csv")
events = pd.read_csv("good_grasp_events.csv")

keywords = [
    "infant",
    "baby",
    "babies",
    "toddler",
    "toddlers",
    "nursery"
]

text = (
    videos["title"].fillna("") + " " +
    videos["description"].fillna("") + " " +
    videos["keywords"].fillna("")
).str.lower()

mask = False
for word in keywords:
    mask = mask | text.str.contains(word, regex=False)

young_videos = videos[mask].copy()

print("\nPotential infant/toddler source videos:")
print(
    young_videos[
        ["video_id", "title", "author", "fps", "resolution"]
    ].to_string(index=False)
)

# Connect clips to those source videos
young_clips = clips[
    clips["video_id"].isin(young_videos["video_id"])
]

# Connect to our good grasp events
young_events = events[
    events["clip"].isin(young_clips["clip"])
].copy()

young_events = young_events.sort_values(
    "mean_visibility",
    ascending=False
)

print("\n==============================")
print("GOOD GRASP EVENTS")
print("==============================")

print("\nNumber of potential infant/toddler source videos:")
print(len(young_videos))

print("\nNumber of corresponding ChildPlay clips:")
print(len(young_clips))

print("\nNumber of good grasp events:")
print(len(young_events))

print("\nTop candidate grasp events:")
print(
    young_events[
        [
            "clip",
            "hand",
            "start_frame",
            "end_frame",
            "n_frames",
            "mean_visibility",
            "split"
        ]
    ].head(30).to_string(index=False)
)

young_events.to_csv(
    "young_child_grasp_candidates.csv",
    index=False
)

young_videos.to_csv(
    "young_child_source_videos.csv",
    index=False
)

print("\nSaved:")
print("young_child_grasp_candidates.csv")
print("young_child_source_videos.csv")
