from src.models.event import Event

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


class OverloadedVehicleDetector:
    def __init__(
        self,
        min_persons: int = 3,
        min_co_motion_frames: int = 5,
        speed_similarity: float = 0.4,
    ):
        self.min_persons = min_persons
        self.min_co_motion_frames = min_co_motion_frames
        self.speed_similarity = speed_similarity
        self._prev: dict[int, tuple[int, int]] = {}
        self._co_motion: dict[tuple[int, int], int] = {}
        self._reported: set[int] = set()

    def process(self, tracks: list, frame_index: int) -> list[Event]:
        events: list[Event] = []
        vehicles = [t for t in tracks if t.class_name in VEHICLE_CLASSES]
        persons = [t for t in tracks if t.class_name == "person"]

        current: dict[int, tuple[int, int]] = {}
        for t in tracks:
            cx = (t.bbox[0] + t.bbox[2]) // 2
            cy = (t.bbox[1] + t.bbox[3]) // 2
            current[t.id] = (cx, cy)

        active_ids = {t.id for t in tracks}
        self._prev = {k: v for k, v in self._prev.items() if k in active_ids}
        self._co_motion = {k: v for k, v in self._co_motion.items()
                           if k[0] in active_ids and k[1] in active_ids}

        for v in vehicles:
            v_cx, v_cy = current[v.id]
            on_board: list[int] = []
            vx1, vy1, vx2, vy2 = v.bbox
            v_margin_x = (vx2 - vx1) * 0.15
            v_margin_y = (vy2 - vy1) * 0.15
            v_bbox_exp = (vx1 - v_margin_x, vy1 - v_margin_y,
                          vx2 + v_margin_x, vy2 + v_margin_y)

            for p in persons:
                pcx, pcy = current[p.id]
                if not (v_bbox_exp[0] <= pcx <= v_bbox_exp[2] and
                        v_bbox_exp[1] <= pcy <= v_bbox_exp[3]):
                    continue

                key = (v.id, p.id)
                prev_v = self._prev.get(v.id)
                prev_p = self._prev.get(p.id)
                if prev_v and prev_p:
                    v_dx = v_cx - prev_v[0]
                    v_dy = v_cy - prev_v[1]
                    p_dx = pcx - prev_p[0]
                    p_dy = pcy - prev_p[1]
                    mag_v = (v_dx * v_dx + v_dy * v_dy) ** 0.5
                    mag_p = (p_dx * p_dx + p_dy * p_dy) ** 0.5
                    if mag_v > 0 and mag_p > 0:
                        dot = v_dx * p_dx + v_dy * p_dy
                        sim = dot / (mag_v * mag_p)
                    else:
                        sim = 0.0
                    if sim > self.speed_similarity:
                        self._co_motion[key] = self._co_motion.get(key, 0) + 1
                    else:
                        self._co_motion[key] = max(0, self._co_motion.get(key, 0) - 1)
                else:
                    self._co_motion[key] = max(0, self._co_motion.get(key, 0))

                threshold_frames = self.min_co_motion_frames if self._co_motion.get(key, 0) > 0 else 0
                if self._co_motion.get(key, 0) >= threshold_frames and threshold_frames > 0:
                    on_board.append(p.id)

            if len(on_board) >= self.min_persons and v.id not in self._reported:
                self._reported.add(v.id)
                events.append(Event(
                    event_type="overloaded_vehicle",
                    track_id=v.id,
                    class_name=v.class_name,
                    location=[int((vx1 + vx2) / 2), int((vy1 + vy2) / 2)],
                    message=f"{v.class_name.title()} #{v.id} overloaded with {len(on_board)} persons",
                    severity="high",
                ))

        self._prev = current
        return events
