from dataclasses import dataclass, field
from pathlib import Path
import cv2
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
    age: int = 0
    embedding: np.ndarray | None = None
    embedding_frame: int = -1
    camera_id: str = ""
    global_id: int = -1
    attributes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "class": self.class_name,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 3),
            "bbox": list(self.bbox),
            "age": self.age,
            "camera_id": self.camera_id,
            "global_id": self.global_id,
            "attributes": self.attributes,
        }


COCO_CLASSES = {
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


REID_MODELS = {
    "x0_25": "osnet_x0_25_msmt17.pt",
    "x0_5": "osnet_x0_5_msmt17.pt",
    "x0_75": "osnet_x0_75_msmt17.pt",
    "x1_0": "osnet_x1_0_msmt17.pt",
    "ain_x1_0": "osnet_ain_x1_0_msmt17.pt",
}


class Tracker:
    def __init__(
        self,
        track_thresh: float = 0.4,
        track_buffer: int = 450,
        match_thresh: float = 0.7,
        track_low_thresh: float = 0.1,
        use_reid: bool = True,
        reid_model: str = "x1_0",
        device: str = "cpu",
        use_cmc: bool = False,
        reid_refresh_interval: int = 50,
        reid_new_track_frames: int = 3,
        camera_id: str = "",
    ):
        self._class_map = COCO_CLASSES
        self._device = device
        self._camera_id = camera_id
        self._reid_refresh_interval = reid_refresh_interval
        self._reid_new_track_frames = reid_new_track_frames

        self._reid_wrapper = None
        if use_reid and BotSort is not None:
            from boxmot.reid import ReID
            _device = torch.device(device)
            reid_name = REID_MODELS.get(reid_model, "osnet_x1_0_msmt17.pt")
            self._reid_wrapper = ReID(reid_name, device=_device, half=False)
            self.tracker = BotSort(
                reid_model=self._reid_wrapper.model,
                track_high_thresh=track_thresh,
                track_low_thresh=track_low_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
                with_reid=True,
                use_cmc=use_cmc,
            )
        else:
            self.tracker = ByteTrack(
                track_thresh=track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
            )

        self._track_ages: dict[int, int] = {}
        self._track_embeddings: dict[int, np.ndarray] = {}
        self._track_embedding_frames: dict[int, int] = {}

    def _compute_embedding(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        if self._reid_wrapper is None:
            return None
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        try:
            emb = self._reid_wrapper.get_features(crop_bgr)
            if emb is not None and emb.numel() > 0:
                return emb.cpu().numpy().flatten()
        except Exception:
            pass
        return None

    @staticmethod
    def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-8)

    def update(self, detections: list, frame: np.ndarray, frame_index: int = 0) -> list[Track]:
        if not detections:
            self.tracker.update(np.empty((0, 6)), frame)
            return []

        dets_np = np.array([
            [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence, d.class_id]
            for d in detections
        ], dtype=np.float32)

        # Pre-compute embeddings for all detections so boxmot skips its own
        det_embs = [self._compute_embedding(frame, d.bbox) for d in detections]
        has_valid = any(e is not None for e in det_embs)
        if has_valid:
            emb_dim = next(e.shape[0] for e in det_embs if e is not None)
            embs_np = np.stack([
                e if e is not None else np.zeros(emb_dim, dtype=np.float32)
                for e in det_embs
            ])
            try:
                raw_tracks = self.tracker.update(dets_np, frame, embs=embs_np)
            except TypeError:
                raw_tracks = self.tracker.update(dets_np, frame)
        else:
            raw_tracks = self.tracker.update(dets_np, frame)

        active_ids = set()
        tracks = []
        for ti, t in enumerate(raw_tracks):
            x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
            track_id = int(t[4])
            conf = float(t[5])
            cls_id = int(t[6])
            active_ids.add(track_id)

            age = self._track_ages.get(track_id, 0) + 1
            self._track_ages[track_id] = age

            is_new = age <= self._reid_new_track_frames
            needs_refresh = (
                is_new
                or (age % self._reid_refresh_interval == 0)
                or (track_id not in self._track_embeddings)
            )

            embedding = self._track_embeddings.get(track_id)
            embedding_frame = self._track_embedding_frames.get(track_id, -1)

            if needs_refresh:
                # Match track to detection by IoU to reuse pre-computed embedding
                matched_emb = None
                for j, d in enumerate(detections):
                    if self._bbox_iou((x1, y1, x2, y2), d.bbox) > 0.3:
                        if det_embs[j] is not None:
                            matched_emb = det_embs[j]
                        break
                if matched_emb is not None:
                    embedding = matched_emb
                    embedding_frame = frame_index
                    self._track_embeddings[track_id] = matched_emb
                    self._track_embedding_frames[track_id] = frame_index
                else:
                    new_emb = self._compute_embedding(frame, (x1, y1, x2, y2))
                    if new_emb is not None:
                        embedding = new_emb
                        embedding_frame = frame_index
                        self._track_embeddings[track_id] = new_emb
                        self._track_embedding_frames[track_id] = frame_index

            tracks.append(Track(
                id=track_id,
                class_id=cls_id,
                class_name=self._class_map.get(cls_id, "unknown"),
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                age=age,
                embedding=embedding,
                embedding_frame=embedding_frame,
                camera_id=self._camera_id,
            ))

        stale_ids = set(self._track_ages.keys()) - active_ids
        for sid in stale_ids:
            self._track_ages.pop(sid, None)
            self._track_embeddings.pop(sid, None)
            self._track_embedding_frames.pop(sid, None)

        return tracks
