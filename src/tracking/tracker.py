from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

try:
    from boxmot.trackers.botsort.botsort import BotSort
except ImportError:
    try:
        from boxmot.trackers.bbox.botsort import BotSort
    except ImportError:
        try:
            from boxmot.trackers import BotSort
        except ImportError:
            BotSort = None

try:
    from boxmot.trackers.bytetrack.bytetrack import ByteTrack
except ImportError:
    try:
        from boxmot.trackers.bbox.bytetrack import ByteTrack
    except ImportError:
        try:
            from boxmot.trackers.bbox.bytetrack.bytetrack import ByteTrack
        except ImportError:
            try:
                from boxmot.trackers import ByteTrack
            except ImportError:
                try:
                    from boxmot import ByteTrack
                except ImportError:
                    import boxmot
                    raise ImportError(
                        f"Could not import ByteTrack or BotSort from boxmot v{boxmot.__version__}. "
                        "Try: pip install -U boxmot>=19"
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
        track_buffer: int = 300,
        match_thresh: float = 0.8,
        track_low_thresh: float = 0.1,
        use_reid: bool = True,
        device: str = "cpu",
    ):
        self._class_map = COCO_CLASSES
        self._device = device

        if use_reid and BotSort is not None:
            from boxmot.reid import ReID
            _device = torch.device(device)
            _reid_model = ReID("osnet_x0_25_msmt17.pt", device=_device, half=False)
            self.tracker = BotSort(
                reid_model=_reid_model.model,
                track_high_thresh=track_thresh,
                track_low_thresh=track_low_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
                with_reid=True,
            )
        else:
            self.tracker = ByteTrack(
                track_thresh=track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
            )

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
