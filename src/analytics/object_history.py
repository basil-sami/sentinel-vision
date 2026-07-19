from dataclasses import dataclass, field


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
        return {
            "id": self.id,
            "class": self.class_name,
            "class_id": self.class_id,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "duration_frames": self.duration_frames,
            "path": self.centroid_path,
        }


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
                )
            self._objects[t.id].add_position(frame_index, t.bbox)
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
