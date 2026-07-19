import cv2
import numpy as np


def draw_zones(
    frame_bgr: np.ndarray,
    zones: list,
    active_zones: set[str] | None = None,
    counters: dict | None = None,
) -> np.ndarray:
    for zone in zones:
        pts = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
        color = _zone_color(zone.zone_type)
        is_active = active_zones and zone.name in active_zones

        if is_active:
            glow_color = (0, 255, 255)
            for thickness in [8, 6, 4]:
                cv2.polylines(frame_bgr, [pts], True, glow_color, thickness)
            overlay = frame_bgr.copy()
            cv2.fillPoly(overlay, [pts], glow_color + (80,))
            frame_bgr = cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0)
        else:
            cv2.polylines(frame_bgr, [pts], True, color, 2)
            overlay = frame_bgr.copy()
            cv2.fillPoly(overlay, [pts], color + (40,))
            frame_bgr = cv2.addWeighted(overlay, 0.3, frame_bgr, 0.7, 0)

        cx = int(np.mean([p[0] for p in zone.polygon]))
        cy = int(np.mean([p[1] for p in zone.polygon]))

        label = zone.name
        if zone.zone_type != "generic":
            label += f" [{zone.zone_type}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bg_color = (0, 255, 255) if is_active else (64, 64, 64)
        cv2.rectangle(frame_bgr, (cx - tw // 2 - 4, cy - th - 4),
                      (cx + tw // 2 + 4, cy + 4), bg_color, -1)
        cv2.putText(frame_bgr, label, (cx - tw // 2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) if is_active else (255, 255, 255), 1)

    if counters:
        y_offset = 30
        cv2.putText(frame_bgr, "Gate Counts:", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        for gate_name, counts in counters.items():
            net = counts["net"]
            net_color = (0, 255, 0) if net >= 0 else (0, 0, 255)
            text = f"{gate_name}: +{counts['entries']}  -{counts['exits']}  (net {net:+d})"
            cv2.putText(frame_bgr, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, net_color, 1)
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


def draw_event_ticker(frame_bgr: np.ndarray, events: list) -> np.ndarray:
    recent = events[-5:]
    y_offset = frame_bgr.shape[0] - 20 * len(recent) - 10
    for ev in recent:
        msg = ev.message[:65] if hasattr(ev, "message") else str(ev)[:65]
        cv2.putText(frame_bgr, msg, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        y_offset += 18
    return frame_bgr


def _zone_color(zone_type: str) -> tuple:
    palette = {
        "restricted": (0, 0, 255),
        "entrance": (0, 200, 0),
        "parking": (0, 200, 200),
        "walkway": (200, 0, 200),
        "generic": (100, 100, 100),
    }
    return palette.get(zone_type, (100, 100, 100))
