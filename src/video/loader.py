import cv2
import numpy as np
from pathlib import Path


class VideoLoader:
    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        self.cap = cv2.VideoCapture(str(self.path))
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._current_frame = 0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def duration(self) -> float:
        return self._frame_count / self._fps if self._fps > 0 else 0.0

    def read_frame(self) -> np.ndarray | None:
        ret, frame = self.cap.read()
        if ret:
            self._current_frame += 1
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def seek(self, frame_index: int) -> bool:
        self._current_frame = frame_index
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        frame = self.read_frame()
        if frame is None:
            raise StopIteration
        return frame

    def get_frame_at_time(self, time_sec: float) -> np.ndarray | None:
        target_frame = int(time_sec * self._fps)
        self.seek(target_frame)
        return self.read_frame()
