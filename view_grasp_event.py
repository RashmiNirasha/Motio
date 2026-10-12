import cv2
import matplotlib.pyplot as plt

clip = "4Y7MMmedX7E_4371-4987"
video_id = "4Y7MMmedX7E"
clip_start = 4371

local_frames = [347, 351, 356, 361, 366]

fig, axes = plt.subplots(1, 5, figsize=(20, 5))

for ax, local_frame in zip(axes, local_frames):

    original_frame = clip_start + local_frame - 1

    path = (
        f"data/ChildPlay-gaze/images/"
        f"{clip}/"
        f"{video_id}_{original_frame}.jpg"
    )

    img = cv2.imread(path)

    if img is None:
        ax.set_title(f"Missing\n{local_frame}")
        ax.axis("off")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ax.imshow(img)
    ax.set_title(
        f"Clip frame {local_frame}\n"
        f"Original {original_frame}"
    )
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    "pilot_grasp_full_scene.png",
    dpi=180,
    bbox_inches="tight"
)

print("Saved pilot_grasp_full_scene.png")
