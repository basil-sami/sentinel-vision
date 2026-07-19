from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ultralytics import YOLO

from src.optimization.tensorrt_export import has_engine, engine_path, export_to_engine


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2

    def to_dict(self) -> dict:
        return {
            "class": self.class_name,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 3),
            "bbox": list(self.bbox),
        }


COCO_TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


MODEL_FAMILIES = {
    "yolo11": {
        "nano": "yolo11n.pt",
        "small": "yolo11s.pt",
        "medium": "yolo11m.pt",
        "large": "yolo11l.pt",
        "xlarge": "yolo11x.pt",
    },
    "yolo12": {
        "nano": "yolo12n.pt",
        "small": "yolo12s.pt",
        "medium": "yolo12m.pt",
        "large": "yolo12l.pt",
        "xlarge": "yolo12x.pt",
    },
    "rtdetr": {
        "nano": "rtdetr-l.pt",
        "large": "rtdetr-l.pt",
        "xlarge": "rtdetr-x.pt",
    },
}


class YOLODetector:
    def __init__(
        self,
        model_family: str = "yolo11",
        model_size: str = "nano",
        device: str = "cpu",
        target_classes: dict[int, str] | None = None,
        use_tensorrt: bool = False,
        tensorrt_half: bool = True,
    ):
        self.device = device
        self.target_classes = target_classes if target_classes is not None else COCO_TARGET_CLASSES

        if use_tensorrt and device.startswith("cuda"):
            if has_engine(model_family, model_size, half=tensorrt_half):
                model_path = engine_path(model_family, model_size, half=tensorrt_half)
            else:
                try:
                    model_path = export_to_engine(
                        model_family=model_family,
                        model_size=model_size,
                        half=tensorrt_half,
                        device=0,
                    )
                except Exception:
                    model_path = None
            if model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
                return

        family = MODEL_FAMILIES.get(model_family, MODEL_FAMILIES["yolo11"])
        model_name = family.get(model_size, "yolo11n.pt")
        self.model = YOLO(model_name)

    def detect(self, image: np.ndarray, conf_threshold: float = 0.5) -> list[Detection]:
        results = self.model.predict(
            image,
            conf=conf_threshold,
            device=self.device,
            verbose=False,
        )
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id not in self.target_classes:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=self.target_classes[class_id],
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                    )
                )
        return detections
