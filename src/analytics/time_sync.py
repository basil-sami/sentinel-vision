import time
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class FrameTimestamp:
    frame_index: int
    utc_time: str
    utc_timestamp: float
    camera_timestamp: float
    processing_timestamp: float


class TimeSync:
    def __init__(self, fps: float = 25.0):
        self._fps = fps
        self._start_time: float | None = None
        self._frame_times: dict[int, FrameTimestamp] = {}

    def start(self):
        self._start_time = time.time()

    def frame_timestamp(self, frame_index: int) -> FrameTimestamp:
        now = time.time()
        if self._start_time is None:
            self._start_time = now
        elapsed = frame_index / self._fps
        utc_ts = self._start_time + elapsed
        utc_dt = datetime.fromtimestamp(utc_ts, tz=timezone.utc)

        ts = FrameTimestamp(
            frame_index=frame_index,
            utc_time=utc_dt.isoformat(),
            utc_timestamp=utc_ts,
            camera_timestamp=elapsed,
            processing_timestamp=now,
        )
        self._frame_times[frame_index] = ts
        return ts

    def get(self, frame_index: int) -> FrameTimestamp | None:
        return self._frame_times.get(frame_index)

    def utc_for_frame(self, frame_index: int) -> str:
        ts = self.get(frame_index)
        return ts.utc_time if ts else ""

    def reset(self):
        self._start_time = None
        self._frame_times.clear()
