import pandas as pd

events = pd.read_csv("verified_child_grasp_events.csv")
clips = pd.read_csv("data/ChildPlay-gaze/clips.csv")
videos = pd.read_csv("data/ChildPlay-gaze/videos.csv")

data = events.merge(
    clips[["clip", "video_id"]],
    on="clip",
    how="left"
)

data = data.merge(
    videos[
        [
            "video_id",
            "title",
            "description",
            "keywords",
            "fps",
            "resolution"
        ]
    ],
    on="video_id",
    how="left"
)

# One row per source video for screening
sources = (
    data[
        [
            "video_id",
            "title",
            "description",
            "fps",
            "resolution"
        ]
    ]
    .drop_duplicates("video_id")
    .sort_values("video_id")
)

print("\nSOURCE VIDEOS REPRESENTED IN THE 95 CHILD EVENTS")
print("=" * 80)

for _, row in sources.iterrows():

    print("\nVIDEO ID:", row["video_id"])
    print("TITLE:", row["title"])
    print("FPS:", row["fps"])
    print("RESOLUTION:", row["resolution"])

    desc = str(row["description"])

    # Print only beginning of description
    print("DESCRIPTION:")
    print(desc[:600].replace("\n", " "))

    n_events = len(
        data[data["video_id"] == row["video_id"]]
    )

    print("GOOD CHILD GRASP EVENTS:", n_events)
    print("-" * 80)

sources.to_csv(
    "source_age_screening.csv",
    index=False
)

print("\nSaved: source_age_screening.csv")
print("Number of source videos:", len(sources))
