from collections import defaultdict

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
from src.analytics.vehicle.attribute_cache import AttributeManager, AttributeState, MAX_PLATE_ATTEMPTS, COLOR_CONSISTENT_NEEDED
from src.analytics.vehicle.candidate_buffer import TopKBuffer
from src.models.event import Event


VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


TEMPORAL_FUSION_FRAMES = 10
TEMP_FUSION_MIN_CONF = 0.3

TOP_K_PLATES = 5
PLATE_HIGH_QUALITY_THRESHOLD = 0.85
OCR_FRAME_INTERVAL = 20


class VehicleAnalyzer:
    def __init__(self, parking_timeout_sec: float = 300.0, plate_read_interval: int = 10):
        self._plate_detector = PlateDetector()
        self._plate_reader = PlateReader()
        self._registry = VehicleRegistry(parking_timeout_sec=parking_timeout_sec)
        self._attrs = AttributeManager()
        self._buffers: dict[int, TopKBuffer] = {}
        self._last_positions: dict[int, tuple[int, int]] = {}
        self._stationary_start: dict[int, int] = {}
        self._plate_buffer: dict[int, list] = defaultdict(list)
        self._reported_plates: set[int] = set()
        self._last_read_frame: dict[int, int] = {}

    def _get_buffer(self, track_id: int) -> TopKBuffer:
        if track_id not in self._buffers:
            self._buffers[track_id] = TopKBuffer(k=TOP_K_PLATES)
        return self._buffers[track_id]

    def process_frame(self, frame, tracks: list, frame_index: int, calibrator) -> list[Event]:
        events = []
        seen_tracks = set()

        for t in tracks:
            if t.class_name not in VEHICLE_CLASSES:
                continue
            seen_tracks.add(t.id)
            attrs = self._attrs.get(t.id)

            cx = (t.bbox[0] + t.bbox[2]) // 2
            cy = (t.bbox[1] + t.bbox[3]) // 2

            # --- Color: 3 consistent observations then lock ---
            if attrs.color.state not in (AttributeState.VERIFIED, AttributeState.LOCKED, AttributeState.FAILED):
                color_info = extract_vehicle_color(frame, t.bbox)
                attrs.color.observations.append(color_info["color"])
                if len(attrs.color.observations) >= COLOR_CONSISTENT_NEEDED:
                    recent = attrs.color.observations[-COLOR_CONSISTENT_NEEDED:]
                    if len(set(recent)) == 1:
                        attrs.color.value = recent[0]
                        attrs.color.confidence = color_info["confidence"]
                        attrs.color.state = AttributeState.LOCKED
                    else:
                        attrs.color.state = AttributeState.FAILED

            # --- Size class: lock after 5 frames ---
            if attrs.size_class.state not in (AttributeState.VERIFIED, AttributeState.LOCKED, AttributeState.FAILED):
                attrs.size_class.observations.append(vehicle_size_class(t.bbox))
                if len(attrs.size_class.observations) >= 5:
                    sizes = attrs.size_class.observations
                    attrs.size_class.value = max(set(sizes), key=sizes.count)
                    attrs.size_class.state = AttributeState.LOCKED

            color_info = {
                "color": attrs.color.value or "unknown",
                "confidence": attrs.color.confidence,
            }
            size_class = attrs.size_class.value or vehicle_size_class(t.bbox)

            # --- Plate: collect top-K candidates, OCR only when needed ---
            plate_state = attrs.plate.state
            if plate_state not in (AttributeState.VERIFIED, AttributeState.LOCKED, AttributeState.FAILED):
                buffer = self._get_buffer(t.id)
                last_read = self._last_read_frame.get(t.id, -1)

                should_detect = False
                if plate_state == AttributeState.UNKNOWN and frame_index - last_read >= OCR_FRAME_INTERVAL:
                    should_detect = True
                elif plate_state == AttributeState.PROCESSING and attrs.plate.attempts < MAX_PLATE_ATTEMPTS:
                    if frame_index - last_read >= OCR_FRAME_INTERVAL:
                        should_detect = True

                if should_detect:
                    self._last_read_frame[t.id] = frame_index
                    attrs.plate.attempts += 1
                    if attrs.plate.state == AttributeState.UNKNOWN:
                        attrs.plate.state = AttributeState.PROCESSING

                    plate_result = self._plate_detector.detect(frame, t.bbox)
                    if plate_result:
                        crop = frame[
                            plate_result["bbox"][1]:plate_result["bbox"][3],
                            plate_result["bbox"][0]:plate_result["bbox"][2],
                        ]
                        if crop.size > 0:
                            buffer.evaluate_and_add(
                                plate_crop=crop,
                                vehicle_crop=frame[t.bbox[1]:t.bbox[3], t.bbox[0]:t.bbox[2]],
                                plate_bbox=plate_result["bbox"],
                                detection_conf=plate_result.get("confidence", 0.5),
                                frame_index=frame_index,
                                frame_area=frame.shape[0] * frame.shape[1],
                            )

                            best = buffer.best()
                            if best and best.score >= PLATE_HIGH_QUALITY_THRESHOLD:
                                self._run_ocr(t.id, best, events, [cx, cy])

                if attrs.plate.attempts >= MAX_PLATE_ATTEMPTS and attrs.plate.state != AttributeState.VERIFIED:
                    if len(buffer) > 0:
                        self._run_ocr(t.id, buffer.best(), events, [cx, cy])
                    else:
                        attrs.plate.state = AttributeState.FAILED

            # --- Speed check ---
            prev_pos = self._last_positions.get(t.id)
            self._last_positions[t.id] = (cx, cy)

            if calibrator and calibrator.is_calibrated and prev_pos:
                speed = calibrator.speed_in_world(
                    [[prev_pos[0], prev_pos[1]], [cx, cy]],
                    fps=25.0,
                )
                if speed > SPEEDING_THRESHOLD_MPS:
                    events.append(speeding_event(t.id, speed, [cx, cy]))

            # --- Stationary / parking ---
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
                            t.id,
                            attrs.plate.value or "",
                            duration_frames / 25.0,
                            [cx, cy],
                        ))
                else:
                    if t.id in self._stationary_start:
                        self._registry.mark_departure(t.id)
                    self._stationary_start.pop(t.id, None)

            # --- Update registry ---
            self._registry.register(
                track_id=t.id,
                plate=attrs.plate.value or "",
                color=color_info["color"],
                vehicle_type=t.class_name,
                size_class=size_class,
            )

        self._attrs.cleanup(seen_tracks)
        stale_buffers = list(set(self._buffers.keys()) - seen_tracks)
        for sid in stale_buffers:
            self._try_final_ocr(sid, events)
            self._buffers.pop(sid, None)
        self._clean_plate_buffers(seen_tracks)
        return events

    def _run_ocr(self, track_id: int, candidate, events: list, location: list[int, int]):
        attrs = self._attrs.get(track_id)
        if candidate is None or candidate.plate_crop is None:
            attrs.plate.state = AttributeState.FAILED
            return

        read_result = self._plate_reader.read(candidate.plate_crop)
        plate_text = read_result.get("plate", "")
        plate_conf = read_result.get("confidence", 0.0)

        if plate_text and plate_conf > 0:
            attrs.plate.value = plate_text
            attrs.plate.confidence = plate_conf
            attrs.plate.state = AttributeState.LOCKED

            if track_id not in self._reported_plates:
                self._reported_plates.add(track_id)
                events.append(plate_read_event(track_id, plate_text, plate_conf, location))

            self._plate_buffer[track_id].append((plate_text, plate_conf))
        else:
            if attrs.plate.attempts >= MAX_PLATE_ATTEMPTS:
                attrs.plate.state = AttributeState.FAILED

    def _try_final_ocr(self, track_id: int, events: list):
        attrs = self._attrs.get(track_id)
        if attrs.plate.state in (AttributeState.VERIFIED, AttributeState.LOCKED, AttributeState.FAILED):
            return
        buffer = self._buffers.get(track_id)
        if buffer and len(buffer) > 0:
            self._run_ocr(track_id, buffer.best(), events, [0, 0])
        else:
            attrs.plate.state = AttributeState.FAILED

    def _fuse_plate(self, track_id: int, current_frame: int) -> tuple[str, float]:
        reads = self._plate_buffer.get(track_id, [])
        reads = [r for r in reads if r[1] >= TEMP_FUSION_MIN_CONF]
        if not reads:
            return ("", 0.0)

        if len(reads) < 3:
            best = max(reads, key=lambda r: r[1])
            return best

        freq: dict[str, list[float]] = {}
        for plate, conf in reads:
            freq.setdefault(plate, []).append(conf)

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
                best = max(buf, key=lambda r: r[1])
                self._registry.register(tid, best[0], "", "", "")
            del self._plate_buffer[tid]

    def get_registry(self) -> VehicleRegistry:
        return self._registry
