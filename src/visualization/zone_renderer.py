import cv2
import numpy as np


def draw_zones(frame_bgr: np.ndarray, zones: list, counters: dict | None = None) -> np.ndarray:
    for zone in zones:
        pts = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
        color = _zone_color(zone.zone_type)
        overlay = frame_bgr.copy()
        cv2.polylines(overlay, [pts], True, color, 2)
        cv2.fillPoly(overlay, [pts], color + (64,))
        frame_bgr = cv2.addWeighted(overlay, 0.4, frame_bgr, 0.6, 0)

        cx = int(np.mean([p[0] for p in zone.polygon]))
        cy = int(np.mean([p[1] for p in zone.polygon]))
        cv2.putText(frame_bgr, zone.name, (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if counters:
        y_offset = 30
        cv2.putText(frame_bgr, "Gate Counts:", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        for gate_name, counts in counters.items():
            text = f"{gate_name}: +{counts['entries']} -{counts['exits']} (net {counts['net']:+d})"
            cv2.putText(frame_bgr, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
            y_offset += 20

    return frame_bgr


def draw_gates(frame_bgr: np.ndarray, gates: list) -> np.ndarray:
    for gate in gates:
        cv2.line(frame_bgr, tuple(gate.p1), tuple(gate.p2), (0, 255, 255), 2)
        mx = (gate.p1[0] + gate.p2[0]) // 2
        my = (gate.p1[1] + gate.p2[1]) // 2
        cv2.putText(frame_bgr, gate.name, (mx - 30, my - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return frame_bgr


def _zone_color(zone_type: str) -> tuple:
    palette = {
        "restricted": (0, 0, 255),
        "entrance": (0, 255, 0),
        "parking": (255, 255, 0),
        "walkway": (255, 0, 255),
    }
    return palette.get(zone_type, (128, 128, 128))
