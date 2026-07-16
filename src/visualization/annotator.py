import cv2
import numpy as np


_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
]


def _id_color(object_id: int) -> tuple[int, int, int]:
    return _COLORS[object_id % len(_COLORS)]


class Annotator:
    def __init__(self, output_path: str, fps: float, width: int, height: int):
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def draw_detections(self, frame: np.ndarray, detections: list, object_id: int | None = None) -> np.ndarray:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = _id_color(object_id) if object_id is not None else (0, 255, 0)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            if object_id is not None:
                label = f"ID {object_id} {label}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame_bgr, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def write_frame(self, frame: np.ndarray):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame_bgr)

    def release(self):
        self.writer.release()
