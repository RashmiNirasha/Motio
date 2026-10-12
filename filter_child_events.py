import pandas as pd
import glob
import os

events = pd.read_csv("young_child_grasp_candidates.csv")

results = []

for _, event in events.iterrows():

    clip = event["clip"]
    pid = event["pid"]

    files = glob.glob(
        f"data/ChildPlay-gaze/annotations/*/{clip}.csv"
    )

    if not files:
        continue

    gaze = pd.read_csv(files[0])

    person = gaze[
        gaze["person_id"] == pid
    ]

    if len(person) == 0:
        continue

    is_child = person["is_child"].mean() > 0.5

    if is_child:
        row = event.to_dict()
        row["verified_child"] = True
        results.append(row)

child_events = pd.DataFrame(results)

print("Original young-child candidates:", len(events))
print("Verified child events:", len(child_events))

child_events = child_events.sort_values(
    "mean_visibility",
    ascending=False
)

print("\nTop verified child grasp events:\n")

print(
    child_events[
        [
            "clip",
            "pid",
            "hand",
            "start_frame",
            "end_frame",
            "mean_visibility",
            "split"
        ]
    ].head(30).to_string(index=False)
)

child_events.to_csv(
    "verified_child_grasp_events.csv",
    index=False
)

print("\nSaved: verified_child_grasp_events.csv")
