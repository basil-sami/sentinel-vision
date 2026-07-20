from src.models.event import Event

CARRYABLE_CLASSES = {
    "backpack", "handbag", "suitcase", "laptop", "cell phone",
    "book", "bottle", "cup", "umbrella", "knife", "baseball bat",
}


class CarryingDetector:
    def __init__(
        self,
        min_overlap_iou: float = 0.05,
        min_co_motion_frames: int = 5,
        speed_similarity: float = 0.5,
    ):
        self.min_overlap_iou = min_overlap_iou
        self.min_co_motion_frames = min_co_motion_frames
        self.speed_similarity = speed_similarity
        self._prev: dict[int, tuple[int, int]] = {}
        self._co_motion: dict[tuple[int, int], int] = {}

    def process(self, tracks: list, frame_index: int) -> list[Event]:
        events: list[Event] = []
        persons = [t for t in tracks if t.class_name == "person"]
        objects = [t for t in tracks if t.class_name in CARRYABLE_CLASSES]

        current: dict[int, tuple[int, int]] = {}
        for t in tracks:
            cx = (t.bbox[0] + t.bbox[2]) // 2
            cy = (t.bbox[1] + t.bbox[3]) // 2
            current[t.id] = (cx, cy)

        # Clean up stale state
        active_ids = {t.id for t in tracks}
        self._prev = {k: v for k, v in self._prev.items() if k in active_ids}
        self._co_motion = {k: v for k, v in self._co_motion.items()
                           if k[0] in active_ids and k[1] in active_ids}

        for p in persons:
            pcx, pcy = current[p.id]
            for o in objects:
                if o.id == p.id:
                    continue
                iou = self._bbox_iou(p.bbox, o.bbox)
                if iou < self.min_overlap_iou:
                    o_cx = (o.bbox[0] + o.bbox[2]) // 2
                    o_cy = (o.bbox[1] + o.bbox[3]) // 2
                    if not self._centroid_inside(p.bbox, o_cx, o_cy):
                        continue

                key = (p.id, o.id)
                prev_p = self._prev.get(p.id)
                prev_o = self._prev.get(o.id)
                ocx, ocy = current[o.id]

                if prev_p and prev_o:
                    p_dx = pcx - prev_p[0]
                    p_dy = pcy - prev_p[1]
                    o_dx = ocx - prev_o[0]
                    o_dy = ocy - prev_o[1]
                    mag_p = (p_dx * p_dx + p_dy * p_dy) ** 0.5
                    mag_o = (o_dx * o_dx + o_dy * o_dy) ** 0.5
                    if mag_p > 0 and mag_o > 0:
                        dot = p_dx * o_dx + p_dy * o_dy
                        sim = dot / (mag_p * mag_o)
                    else:
                        sim = 0.0
                    if sim > self.speed_similarity:
                        self._co_motion[key] = self._co_motion.get(key, 0) + 1
                    else:
                        self._co_motion[key] = max(0, self._co_motion.get(key, 0) - 1)
                else:
                    self._co_motion[key] = max(0, self._co_motion.get(key, 0))

                if self._co_motion.get(key, 0) >= self.min_co_motion_frames:
                    events.append(Event(
                        event_type="person_carrying",
                        track_id=p.id,
                        class_name=o.class_name,
                        location=[pcx, pcy],
                        message=f"Person #{p.id} carrying {o.class_name} #{o.id}",
                        severity="medium",
                    ))
                    self._co_motion[key] -= self.min_co_motion_frames

        self._prev = current
        return events

    @staticmethod
    def _bbox_iou(a: tuple, b: tuple) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    @staticmethod
    def _centroid_inside(bbox: tuple, cx: int, cy: int) -> bool:
        x1, y1, x2, y2 = bbox
        margin_x = (x2 - x1) * 0.1
        margin_y = (y2 - y1) * 0.1
        return (x1 - margin_x <= cx <= x2 + margin_x and
                y1 - margin_y <= cy <= y2 + margin_y)
