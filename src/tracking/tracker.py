from dataclasses import dataclass
import importlib
import numpy as np


_BYTE_BACKENDS = [
    "boxmot.trackers.bytetrack.bytetrack:ByteTrack",
    "boxmot.trackers.bbox.bytetrack:ByteTrack",
    "boxmot.trackers.bbox.bytetrack.bytetrack:ByteTrack",
    "boxmot.trackers.bbox:ByteTrack",
    "boxmot.trackers:ByteTrack",
    "boxmot:ByteTrack",
]
_BYTETRACK_CLS = None
for _path in _BYTE_BACKENDS:
    try:
        mod_path, cls_name = _path.rsplit(":", 1)
        mod = importlib.import_module(mod_path)
        _BYTETRACK_CLS = getattr(mod, cls_name)
        break
    except (ImportError, AttributeError):
        continue
if _BYTETRACK_CLS is None:
    raise ImportError(
        f"Could not import ByteTrack from any known path: {_BYTE_BACKENDS}. "
        f"Current boxmot version: {importlib.import_module('boxmot').__version__}"
    )


@dataclass
class Track:
    id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]

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
        self.tracker = _BYTETRACK_CLS(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
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

        if raw_tracks.shape[0] == 0:
            return []

        tracks = []
        for t in raw_tracks:
            x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
            track_id = int(t[4])
            conf = float(t[5])
            cls_id = int(t[6])
            tracks.append(Track(
                id=track_id,
                class_id=cls_id,
                class_name=self._class_map.get(cls_id, "unknown"),
                confidence=conf,
                bbox=(x1, y1, x2, y2),
            ))
        return tracks
