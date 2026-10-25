import argparse
import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from observe.detector import ObjectDetector


# --------------------------------------------------
# Try the object detector on photos, no camera or server needed.
#
#   python mvp/observe/test_detector.py photo.jpg
#   python mvp/observe/test_detector.py my_toy_photos/ --save annotated/
#
# Prints the objects found in each image with confidence, and
# the time per image. With --save, writes copies with boxes drawn.
# --------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("images", nargs="+", help="image files or folders")
parser.add_argument("--save", help="folder for annotated copies")
parser.add_argument("--device", help="mps or cpu (default: mps if available)")
parser.add_argument("--min-confidence", type=float, default=0.25)
args = parser.parse_args()

paths = []

for item in args.images:

    if os.path.isdir(item):
        for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG", "heic"):
            paths += sorted(glob.glob(os.path.join(item, f"*.{ext}")))
    else:
        paths.append(item)

if not paths:
    sys.exit("No images found.")

print("Loading detector (first run downloads the model, ~400 MB)...")

detector = ObjectDetector(device=args.device, min_confidence=args.min_confidence)

print(f"Device: {detector.device}\n")

times = []

for path in paths:

    image = cv2.imread(path)

    if image is None:
        print(f"{path}: could not read")
        continue

    detections = detector.detect(image)
    times.append(detector.last_ms)

    found = ", ".join(f"{d.label} {d.confidence:.2f}" for d in detections) or "nothing"
    print(f"{os.path.basename(path)}  ({detector.last_ms:.0f} ms): {found}")

    if args.save:

        os.makedirs(args.save, exist_ok=True)

        h, w = image.shape[:2]

        for d in detections:

            x1, y1, x2, y2 = (int(d.box[0] * w), int(d.box[1] * h), int(d.box[2] * w), int(d.box[3] * h))

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), max(2, w // 400))
            cv2.putText(
                image, f"{d.label} {d.confidence:.2f}", (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.5, w / 1600), (0, 220, 255), 2
            )

        cv2.imwrite(os.path.join(args.save, os.path.basename(path)), image)

if len(times) > 1:

    # The first image includes warm-up.
    steady = sorted(times[1:])
    print(f"\nTime per image after warm-up: median {steady[len(steady) // 2]:.0f} ms")
