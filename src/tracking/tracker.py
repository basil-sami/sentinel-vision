from dataclasses import dataclass
import numpy as np
from boxmot import ByteTrack as _ByteTrack


@dataclass
class Track:
    id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "class": self.class_name,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 3),
            "bbox": list(self.bbox),
        }


COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck",
}


class Tracker:
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ):
        self.tracker = _ByteTrack(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
        )
        self._class_map = COCO_CLASSES

    def update(self, detections: list, frame: np.ndarray) -> list[Track]:
        if not detections:
            self.tracker.update(np.empty((0, 5)), frame)
            return []

        dets_np = np.array([
            [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence]
            for d in detections
        ], dtype=float)

        raw_tracks = self.tracker.update(dets_np, frame)

        tracks = []
        for t in raw_tracks:
            x1, y1, x2, y2, track_id, conf = t[:6]
            cls_id = int(t[6]) if len(t) > 6 else 0
            tracks.append(Track(
                id=int(track_id),
                class_id=cls_id,
                class_name=self._class_map.get(cls_id, "unknown"),
                confidence=float(conf),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
            ))
        return tracks
