import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


class EvidenceCapture:
    def __init__(self, output_dir: str, fps: float, width: int, height: int, pre_frames: int = 30, post_frames: int = 30):
        self._output_dir = Path(output_dir) / "events"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._fps = fps
        self._width = width
        self._height = height
        self._pre_frames = pre_frames
        self._post_frames = post_frames
        self._frame_buffer: list[np.ndarray] = []
        self._pending_clips: list[dict] = []
        self._captured_ids: set[str] = set()

    def add_frame(self, frame_bgr: np.ndarray):
        self._frame_buffer.append(frame_bgr)
        if len(self._frame_buffer) > self._pre_frames + self._post_frames:
            self._frame_buffer.pop(0)

    def capture_for_event(self, event_type: str, track_id: int, metadata: dict):
        clip_id = f"{event_type}_id{track_id}"
        dedup_key = f"{event_type}_{track_id}"
        if dedup_key in self._captured_ids:
            return
        self._captured_ids.add(dedup_key)

        if len(self._frame_buffer) < self._pre_frames + 1:
            return

        clip_dir = self._output_dir / clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)

        frames_to_save = self._frame_buffer[-(self._pre_frames + self._post_frames):]
        clip_path = str(clip_dir / "event.mp4")

        out = cv2.VideoWriter(
            clip_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self._fps,
            (self._width, self._height),
        )
        for f in frames_to_save:
            out.write(f)
        out.release()
        self._try_reencode(clip_path)

        metadata_path = clip_dir / "metadata.json"
        meta = {
            "event": event_type,
            "track_id": track_id,
            "fps": self._fps,
            "frames": len(frames_to_save),
            "duration_sec": round(len(frames_to_save) / self._fps, 1),
            **metadata,
        }
        metadata_path.write_text(json.dumps(meta, indent=2))

    def list_captures(self) -> list[dict]:
        results = []
        for d in sorted(self._output_dir.iterdir()):
            if d.is_dir():
                meta_path = d / "metadata.json"
                if meta_path.exists():
                    results.append(json.loads(meta_path.read_text()))
        return results

    @staticmethod
    def _try_reencode(path: str):
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", path,
                 "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                 path + ".tmp"],
                capture_output=True, check=True,
            )
            shutil.move(path + ".tmp", path)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
