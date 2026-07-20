from collections import defaultdict
from dataclasses import dataclass, field

from src.analytics.vehicle.plate_detector import PlateDetector
from src.analytics.vehicle.plate_reader import PlateReader
from src.analytics.vehicle.attributes import extract_vehicle_color, vehicle_size_class
from src.analytics.vehicle.registry import VehicleRegistry
from src.analytics.vehicle.events import (
    speeding_event,
    parking_event,
    plate_read_event,
    repeat_visitor_event,
    SPEEDING_THRESHOLD_MPS,
)
from src.models.event import Event


VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

TEMPORAL_FUSION_FRAMES = 10
TEMP_FUSION_MIN_CONF = 0.3


@dataclass
class _PlateRead:
    plate: str
    confidence: float
    frame: int


class VehicleAnalyzer:
    def __init__(self, parking_timeout_sec: float = 300.0, plate_read_interval: int = 10):
        self._plate_detector = PlateDetector()
        self._plate_reader = PlateReader()
        self._registry = VehicleRegistry(parking_timeout_sec=parking_timeout_sec)
        self._last_positions: dict[int, tuple[int, int]] = {}
        self._stationary_start: dict[int, int] = {}
        self._reported_plates: set[int] = set()
        self._plate_buffer: dict[int, list[_PlateRead]] = defaultdict(list)
        self._plate_read_interval = plate_read_interval
        self._last_read_frame: dict[int, int] = {}

    def process_frame(
        self,
        frame,
        tracks: list,
        frame_index: int,
        calibrator,
    ) -> list[Event]:
        events = []
        seen_tracks = set()

        for t in tracks:
            if t.class_name not in VEHICLE_CLASSES:
                continue
            seen_tracks.add(t.id)

            cx = (t.bbox[0] + t.bbox[2]) // 2
            cy = (t.bbox[1] + t.bbox[3]) // 2

            color_info = extract_vehicle_color(frame, t.bbox)
            size_class = vehicle_size_class(t.bbox)

            plate_text = ""
            plate_conf = 0.0
            last_read = self._last_read_frame.get(t.id, -1)
            if frame_index - last_read >= self._plate_read_interval:
                self._last_read_frame[t.id] = frame_index
                plate_result = self._plate_detector.detect(frame, t.bbox)
                if plate_result:
                    crop = frame[
                        plate_result["bbox"][1]:plate_result["bbox"][3],
                        plate_result["bbox"][0]:plate_result["bbox"][2],
                    ]
                    if crop.size > 0:
                        read_result = self._plate_reader.read(crop)
                        plate_text = read_result.get("plate", "")
                        plate_conf = read_result.get("confidence", 0.0)

                        if plate_text and plate_conf > 0:
                            self._plate_buffer[t.id].append(
                                _PlateRead(plate=plate_text, confidence=plate_conf, frame=frame_index)
                            )

            fused_plate, fused_conf = self._fuse_plate(t.id, frame_index)
            if fused_plate and t.id not in self._reported_plates:
                self._reported_plates.add(t.id)
                events.append(plate_read_event(t.id, fused_plate, fused_conf, [cx, cy]))

            rec = self._registry.register(
                track_id=t.id,
                plate=fused_plate or plate_text,
                color=color_info["color"],
                vehicle_type=t.class_name,
                size_class=size_class,
            )

            prev_pos = self._last_positions.get(t.id)
            self._last_positions[t.id] = (cx, cy)

            if calibrator and calibrator.is_calibrated and prev_pos:
                speed = calibrator.speed_in_world(
                    [[prev_pos[0], prev_pos[1]], [cx, cy]],
                    fps=25.0,
                )
                if speed > SPEEDING_THRESHOLD_MPS:
                    events.append(speeding_event(t.id, speed, [cx, cy]))

            if prev_pos:
                dx = cx - prev_pos[0]
                dy = cy - prev_pos[1]
                moved = (dx * dx + dy * dy) ** 0.5
                if moved < 5:
                    if t.id not in self._stationary_start:
                        self._stationary_start[t.id] = frame_index
                    duration_frames = frame_index - self._stationary_start[t.id]
                    if duration_frames == 150:
                        self._registry.mark_parking(t.id)
                        events.append(parking_event(
                            t.id, fused_plate or plate_text, duration_frames / 25.0, [cx, cy]
                        ))
                else:
                    if t.id in self._stationary_start:
                        self._registry.mark_departure(t.id)
                    self._stationary_start.pop(t.id, None)

        self._clean_plate_buffers(seen_tracks)
        return events

    def _fuse_plate(self, track_id: int, current_frame: int) -> tuple[str, float]:
        reads = self._plate_buffer.get(track_id, [])
        reads = [r for r in reads if r.confidence >= TEMP_FUSION_MIN_CONF]
        if not reads:
            return ("", 0.0)

        if len(reads) < 3:
            best = max(reads, key=lambda r: r.confidence)
            return (best.plate, best.confidence)

        freq: dict[str, list[float]] = {}
        for r in reads:
            freq.setdefault(r.plate, []).append(r.confidence)

        best_plate = ""
        best_score = 0.0
        for plate, confs in freq.items():
            avg_conf = sum(confs) / len(confs)
            count_score = len(confs) / max(len(reads), 1)
            score = avg_conf * (0.6 + 0.4 * count_score)
            if score > best_score:
                best_score = score
                best_plate = plate

        return (best_plate, round(best_score, 3))

    def _clean_plate_buffers(self, active_tracks: set[int]):
        stale = list(self._plate_buffer.keys() - active_tracks)
        for tid in stale:
            buf = self._plate_buffer.get(tid, [])
            if buf:
                most_recent = max(buf, key=lambda r: r.frame)
                if most_recent.plate:
                    self._registry.register(tid, most_recent.plate, "", "", "")
            del self._plate_buffer[tid]

    def get_registry(self) -> VehicleRegistry:
        return self._registry
