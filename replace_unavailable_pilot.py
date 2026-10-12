import pandas as pd

pilot = pd.read_csv("infant_pilot_20.csv")
events = pd.read_csv("verified_child_grasp_events.csv")
clips = pd.read_csv("data/ChildPlay-gaze/clips.csv")

events = events.merge(
    clips[["clip", "video_id", "fps", "resolution"]],
    on="clip",
    how="left"
)

# Remove unavailable YouTube source
pilot = pilot[
    pilot["video_id"] != "w6oRoyTBmU4"
].copy()

print("After removing unavailable source:", len(pilot))

# Replacement sources
replacement_sources = [
    "Dai13S2XbsI",
    "UUwGLDeYN8c"
]

candidates = events[
    events["video_id"].isin(replacement_sources)
].copy()

# Don't duplicate an event already present
candidates = candidates[
    ~candidates["clip"].isin(pilot["clip"])
]

candidates = candidates.sort_values(
    "mean_visibility",
    ascending=False
)

# Prefer one event from each replacement source
replacement_rows = []

for source in replacement_sources:

    source_events = candidates[
        candidates["video_id"] == source
    ]

    if len(source_events) > 0:
        replacement_rows.append(
            source_events.iloc[0]
        )

replacements = pd.DataFrame(replacement_rows)

print("\nREPLACEMENTS:")
print(
    replacements[
        [
            "video_id",
            "clip",
            "pid",
            "hand",
            "start_frame",
            "end_frame",
            "mean_visibility",
            "split"
        ]
    ].to_string(index=False)
)

# Match pilot annotation columns
for col in pilot.columns:
    if col not in replacements.columns:
        replacements[col] = ""

replacements["age_tier"] = "B"

updated = pd.concat(
    [
        pilot,
        replacements[pilot.columns]
    ],
    ignore_index=True
)

updated.to_csv(
    "infant_pilot_20.csv",
    index=False
)

print("\nUpdated pilot size:", len(updated))

print("\nCounts by source:")
print(updated["video_id"].value_counts())

print("\nSaved updated infant_pilot_20.csv")
