import base64

import numpy as np


@dataclass
class FrameEntry:
    frame: int
    bbox: tuple[int, int, int, int]
    cx: int
    cy: int


@dataclass
class ObjectRecord:
    id: int
    class_name: str
    class_id: int
    first_frame: int
    last_frame: int
    positions: list[tuple[int, int]] = field(default_factory=list)
    bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    camera_id: str = ""
    global_id: int = -1
    embedding: np.ndarray | None = None

    def add_position(self, frame: int, bbox: tuple[int, int, int, int]):
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        self.last_frame = frame
        self.positions.append((cx, cy))
        self.bboxes.append(bbox)

    @property
    def centroid_path(self) -> list[list[int]]:
        return [[x, y] for x, y in self.positions]

    @property
    def duration_frames(self) -> int:
        return self.last_frame - self.first_frame + 1

    def to_dict(self) -> dict:
        emb_b64 = None
        if self.embedding is not None:
            emb_b64 = base64.b64encode(self.embedding.tobytes()).decode()
        return {
            "id": self.id,
            "class": self.class_name,
            "class_id": self.class_id,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "duration_frames": self.duration_frames,
            "path": self.centroid_path,
            "camera_id": self.camera_id,
            "global_id": self.global_id,
            "embedding_b64": emb_b64,
        }

    @staticmethod
    def from_dict(d: dict) -> "ObjectRecord":
        emb = None
        if d.get("embedding_b64"):
            emb = np.frombuffer(base64.b64decode(d["embedding_b64"]), dtype=np.float32).copy()
        return ObjectRecord(
            id=d["id"],
            class_name=d["class"],
            class_id=d.get("class_id", -1),
            first_frame=d["first_frame"],
            last_frame=d["last_frame"],
            positions=[(p[0], p[1]) for p in d.get("path", [])],
            camera_id=d.get("camera_id", ""),
            global_id=d.get("global_id", -1),
            embedding=emb,
        )


class ObjectHistory:
    def __init__(self):
        self._objects: dict[int, ObjectRecord] = {}

    def update(self, tracks: list, frame_index: int):
        seen_ids = set()
        for t in tracks:
            if t.id not in self._objects:
                self._objects[t.id] = ObjectRecord(
                    id=t.id,
                    class_name=t.class_name,
                    class_id=t.class_id,
                    first_frame=frame_index,
                    last_frame=frame_index,
                    camera_id=getattr(t, "camera_id", ""),
                    global_id=getattr(t, "global_id", -1),
                )
            rec = self._objects[t.id]
            rec.add_position(frame_index, t.bbox)
            if t.embedding is not None:
                rec.embedding = t.embedding.copy()
            seen_ids.add(t.id)

    @property
    def tracked_ids(self) -> set[int]:
        return set(self._objects.keys())

    def get(self, object_id: int) -> ObjectRecord | None:
        return self._objects.get(object_id)

    def all(self) -> list[ObjectRecord]:
        return list(self._objects.values())

    def summary(self) -> dict:
        return {
            "total_objects_tracked": len(self._objects),
            "by_class": self._count_by_class(),
        }

    def _count_by_class(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for obj in self._objects.values():
            counts[obj.class_name] = counts.get(obj.class_name, 0) + 1
        return counts

    def export(self) -> list[dict]:
        return [obj.to_dict() for obj in self._objects.values()]
