from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ultralytics import YOLO


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
}


class YOLODetector:
    def __init__(self, model_size: str = "nano", device: str = "cpu"):
        model_map = {
            "nano": "yolo11n.pt",
            "small": "yolo11s.pt",
            "medium": "yolo11m.pt",
        }
        model_name = model_map.get(model_size, "yolo11n.pt")
        self.model = YOLO(model_name)
        self.device = device
        self.target_classes = COCO_TARGET_CLASSES

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
