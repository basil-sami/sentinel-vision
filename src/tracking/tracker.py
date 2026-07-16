from dataclasses import dataclass
import numpy as np


_BYTE_BACKENDS = [
    "boxmot.trackers.ByteTrack",
    "boxmot.trackers.bbox.bytetrack.ByteTrack",
    "boxmot.ByteTrack",
]
_BYTETRACK_CLS = None
for _path in _BYTE_BACKENDS:
    try:
        *mod_parts, cls_name = _path.rsplit(".", 1)
        mod = __import__(".".join(mod_parts), fromlist=[cls_name])
        _BYTETRACK_CLS = getattr(mod, cls_name)
        break
    except (ImportError, AttributeError):
        continue
if _BYTETRACK_CLS is None:
    raise ImportError(
        f"Could not import ByteTrack from any known path: {_BYTE_BACKENDS}. "
        "Try: pip install -U boxmot>=22"
    )


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
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        track_buffer: int = 30,
    ):
        self.tracker = _BYTETRACK_CLS(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            track_buffer=track_buffer,
        )
        self._class_map = COCO_CLASSES

    def update(self, detections: list, frame: np.ndarray) -> list[Track]:
        if not detections:
            self.tracker.update(np.empty((0, 6)), frame)
            return []

        dets_np = np.array([
            [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence, d.class_id]
            for d in detections
        ], dtype=np.float32)

        raw_tracks = self.tracker.update(dets_np, frame)

        tracks = []
        for t in raw_tracks:
            x1, y1, x2, y2, track_id, conf, cls_id = map(int, t[:7])
            tracks.append(Track(
                id=int(track_id),
                class_id=cls_id,
                class_name=self._class_map.get(cls_id, "unknown"),
                confidence=float(conf),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
            ))
        return tracks
