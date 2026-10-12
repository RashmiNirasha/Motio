import pandas as pd
import glob
import os

base = "data/childplay_hand/data/annotations"

all_dfs = []

for split in ["train", "val", "test"]:
    files = glob.glob(f"{base}/{split}/*.csv")

    print(f"{split}: {len(files)} CSV files")

    for file in files:
        df = pd.read_csv(file)
        df["split"] = split
        df["source_file"] = os.path.basename(file)
        all_dfs.append(df)

data = pd.concat(all_dfs, ignore_index=True)

print("\n==============================")
print("DATASET SUMMARY")
print("==============================")

print("\nTotal annotation rows:")
print(len(data))

print("\nColumns:")
print(data.columns.tolist())

print("\nUnique interaction labels:")
print(data["interaction"].dropna().unique())

print("\nInteraction counts:")
print(data["interaction"].value_counts(dropna=False))

print("\nHand sides:")
print(data["hand"].value_counts(dropna=False))

print("\nHand visibility statistics:")
print(data["h_visible"].describe())

# ---- GRASP ANALYSIS ----

grasp = data[
    data["interaction"]
    .astype(str)
    .str.lower()
    .str.contains("grasp", na=False)
].copy()

print("\n==============================")
print("GRASP ANALYSIS")
print("==============================")

print("\nNumber of grasp annotation rows:")
print(len(grasp))

print("\nNumber of unique clips containing grasp:")
print(grasp["source_file"].nunique())

print("\nGrasp rows by split:")
print(grasp["split"].value_counts())

print("\nGrasp hand visibility:")
print(grasp["h_visible"].describe())

high_visibility = grasp[grasp["h_visible"] >= 0.8]

print("\nGrasp rows with hand visibility >= 0.8:")
print(len(high_visibility))

print("\nExample grasp annotations:")
cols = [
    "clip",
    "frame",
    "pid",
    "hand",
    "h_visible",
    "interaction",
    "source_file",
    "split"
]

available_cols = [c for c in cols if c in grasp.columns]

print(
    grasp[available_cols]
    .head(20)
    .to_string(index=False)
)

grasp.to_csv("grasp_annotations.csv", index=False)

print("\nSaved all grasp rows to:")
print("grasp_annotations.csv")
