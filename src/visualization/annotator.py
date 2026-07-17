import subprocess
import cv2
import numpy as np
from pathlib import Path


_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
]


def _id_color(object_id: int) -> tuple[int, int, int]:
    return _COLORS[object_id % len(_COLORS)]


class Annotator:
    def __init__(self, output_path: str, fps: float, width: int, height: int):
        self.output_path = Path(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        self.frame_dir = self.output_path.with_suffix("")
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self._frame_index = 0

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

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: list,
        history: object,
        trail_length: int = 50,
    ) -> np.ndarray:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            color = _id_color(track.id)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            rec = history.get(track.id)
            if rec and len(rec.positions) > 1:
                trail = rec.positions[-trail_length:]
                for j in range(1, len(trail)):
                    alpha = j / len(trail)
                    trail_color = (
                        int(color[0] * alpha),
                        int(color[1] * alpha),
                        int(color[2] * alpha),
                    )
                    cv2.line(frame_bgr, trail[j - 1], trail[j], trail_color, 2)

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track.id} {track.class_name} {track.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame_bgr, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def write_frame(self, frame: np.ndarray):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_path = self.frame_dir / f"{self._frame_index:08d}.png"
        cv2.imwrite(str(frame_path), frame_bgr)
        self._frame_index += 1

    def probe(self, path: Path | None = None) -> dict | None:
        path = path or self.output_path
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_format", "-show_streams",
                 "-of", "json", str(path)],
                capture_output=True, text=True, check=True,
            )
            import json
            info = json.loads(result.stdout)
            print(f"\n--- Probe: {path.name} ---")
            if "format" in info:
                fmt = info["format"]
                print(f"  Format: {fmt.get('format_name')}  duration: {fmt.get('duration','?')}s")
            for s in info.get("streams", []):
                print(f"  Stream #{s['index']}: {s.get('codec_type')}  "
                      f"codec: {s.get('codec_name')}  "
                      f"{s.get('width','?')}x{s.get('height','?')}  "
                      f"fps: {s.get('r_frame_rate','?')}")
            print("---\n")
            return info
        except Exception as e:
            print(f"  Probe failed: {e}")
            return None

    def _encode_ffmpeg(self):
        print(f"Encoding {self._frame_index} frames via ffmpeg...")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(self.fps),
            "-i", str(self.frame_dir / "%08d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "23",
            str(self.output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    def _encode_opencv(self):
        print(f"Encoding {self._frame_index} frames via OpenCV...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps,
                                 (self.width, self.height))
        if not writer.isOpened():
            raise RuntimeError("OpenCV VideoWriter failed to open")
        for i in range(self._frame_index):
            frame_path = self.frame_dir / f"{i:08d}.png"
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                writer.write(frame)
        writer.release()

    def release(self):
        if self._frame_index == 0:
            return
        try:
            self._encode_ffmpeg()
        except (FileNotFoundError, subprocess.CalledProcessError):
            self._encode_opencv()
        self.probe()
        print(f"Raw frames kept in: {self.frame_dir}/")
        print(f"Video saved to: {self.output_path}")
