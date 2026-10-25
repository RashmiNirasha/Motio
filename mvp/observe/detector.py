import os
import time
from dataclasses import dataclass


# --------------------------------------------------
# Play-object detection with a pretrained open-vocabulary
# detector (YOLO-World via Ultralytics). Nothing is trained:
# the detector is told which words to look for.
#
# Several prompts can map to one object name, because the
# detector responds differently to different wordings
# ("toy block" vs "building block"). Boxes from synonyms are
# merged (class-agnostic NMS) and reported under one name.
#
# Frames are processed in memory only; nothing is saved.
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "mvp", "models_detect")

DEFAULT_WEIGHTS = "yolov8s-worldv2.pt"

# object name -> prompts the detector is given
PLAY_OBJECTS = {
    "block": ["building block", "toy block", "wooden block", "lego brick"],
    "crayon": ["crayon"],
    "pencil": ["pencil", "colored pencil"],
    "pen": ["pen", "marker pen"],
    "ball": ["ball", "toy ball"],
    "cup": ["cup", "plastic cup", "stacking cup"],
    "bowl": ["bowl"],
    "container": ["container", "box", "basket"],
    "toy car": ["toy car", "toy truck"],
    "book": ["book", "picture book"],
    "stuffed toy": ["teddy bear", "stuffed toy", "plush toy"],
    "doll": ["doll"],
    "puzzle piece": ["puzzle piece", "jigsaw puzzle"],
    "spoon": ["spoon"],
    "bottle": ["bottle"],
    "play dough": ["play dough", "modeling clay"],
    "ring stacker": ["stacking rings", "ring stacker toy"],
    "toy": ["toy"],
}

# Chosen by looking at ChildPlay scenes: below ~0.25 most boxes
# were wrong (e.g. a bag as "toy car" at 0.17). The generic "toy"
# prompt catches play objects outside the list but is less reliable.
CONFIDENCE = {
    "toy": 0.30,
}

DEFAULT_CONFIDENCE = 0.25


@dataclass
class Detection:

    label: str
    prompt: str
    confidence: float
    # normalised 0..1: x1, y1, x2, y2
    box: tuple

    def to_dict(self):

        return {
            "label": self.label,
            "prompt": self.prompt,
            "confidence": round(self.confidence, 3),
            "box": [round(v, 4) for v in self.box]
        }


def best_device():

    import torch

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


class ObjectDetector:

    def __init__(
        self,
        objects=None,
        weights=DEFAULT_WEIGHTS,
        device=None,
        image_size=640,
        min_confidence=DEFAULT_CONFIDENCE
    ):

        from ultralytics import YOLOWorld

        self.objects = objects or PLAY_OBJECTS
        self.image_size = image_size
        self.min_confidence = min_confidence
        self.device = device or best_device()

        os.makedirs(WEIGHTS_DIR, exist_ok=True)

        # Ultralytics downloads known weights to the given path
        # on first use.
        self.model = YOLOWorld(os.path.join(WEIGHTS_DIR, weights))

        self.prompts = []
        self.prompt_label = {}

        for label, prompts in self.objects.items():
            for prompt in prompts:
                self.prompts.append(prompt)
                self.prompt_label[prompt] = label

        # Encodes the prompts with CLIP (downloaded on first use).
        self.model.set_classes(self.prompts)

        self.last_ms = None

    def detect(self, image):

        # image: numpy array, BGR (OpenCV) or RGB PIL image

        start = time.perf_counter()

        result = self.model.predict(
            image,
            imgsz=self.image_size,
            conf=self.min_confidence,
            iou=0.5,
            agnostic_nms=True,
            device=self.device,
            verbose=False
        )[0]

        self.last_ms = (time.perf_counter() - start) * 1000

        detections = []

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        xyxyn = boxes.xyxyn.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for box, confidence, class_index in zip(xyxyn, confidences, classes):

            prompt = self.prompts[class_index]
            label = self.prompt_label[prompt]

            if confidence < CONFIDENCE.get(label, self.min_confidence):
                continue

            detections.append(
                Detection(
                    label=label,
                    prompt=prompt,
                    confidence=float(confidence),
                    box=tuple(float(v) for v in box)
                )
            )

        detections.sort(key=lambda d: -d.confidence)

        return detections
