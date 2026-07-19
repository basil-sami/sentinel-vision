import numpy as np


class IdentityConfidence:
    def __init__(self):
        self._track_metrics: dict[int, dict] = {}

    def update(self, track_id: int, bbox: tuple[int, int, int, int], confidence: float, frame: int):
        if track_id not in self._track_metrics:
            self._track_metrics[track_id] = {
                "first_frame": frame,
                "last_frame": frame,
                "confidence_sum": 0.0,
                "confidence_count": 0,
                "min_confidence": 1.0,
                "appearances": [],
                "bboxes": [],
                "occlusion_count": 0,
                "last_bbox": bbox,
            }
        m = self._track_metrics[track_id]
        m["last_frame"] = frame
        m["confidence_sum"] += confidence
        m["confidence_count"] += 1
        m["min_confidence"] = min(m["min_confidence"], confidence)
        m["appearances"].append(frame)
        m["bboxes"].append(bbox)

        if confidence < 0.3:
            m["occlusion_count"] += 1

    def get_confidence(self, track_id: int) -> float:
        m = self._track_metrics.get(track_id)
        if not m or m["confidence_count"] == 0:
            return 0.0
        return round(m["confidence_sum"] / m["confidence_count"], 3)

    def get_metrics(self, track_id: int) -> dict:
        m = self._track_metrics.get(track_id)
        if not m:
            return {
                "confidence": 0.0,
                "stability": 0.0,
                "appearance_variance": 0.0,
                "occlusion_count": 0,
                "duration_frames": 0,
            }
        avg_conf = m["confidence_sum"] / m["confidence_count"]
        duration = m["last_frame"] - m["first_frame"] + 1

        appearance_var = 0.0
        if len(m["bboxes"]) >= 2:
            centers = []
            for bx in m["bboxes"]:
                cx = (bx[0] + bx[2]) / 2
                cy = (bx[1] + bx[3]) / 2
                centers.append([cx, cy])
            centers = np.array(centers)
            appearance_var = float(np.var(centers, axis=0).mean())

        stability = round(1.0 - (m["occlusion_count"] / max(len(m["appearances"]), 1)), 3)

        return {
            "confidence": round(avg_conf, 3),
            "stability": stability,
            "appearance_variance": round(appearance_var, 1),
            "occlusion_count": m["occlusion_count"],
            "duration_frames": duration,
        }
