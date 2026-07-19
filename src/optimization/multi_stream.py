import json
import signal
import sys
import time
from multiprocessing import Process, Queue, Event as MP_Event
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


class CameraWorker(Process):
    def __init__(
        self,
        camera_id: int,
        video_path: str,
        output_dir: str,
        event_queue: Queue,
        stop_event: MP_Event,
        model_family: str = "yolo11",
        model_size: str = "nano",
        conf_threshold: float = 0.4,
        device: str = "cuda",
        max_frames: int | None = None,
        use_tensorrt: bool = False,
        **pipeline_kwargs,
    ):
        super().__init__()
        self._cam_id = camera_id
        self._video_path = video_path
        self._output_dir = Path(output_dir)
        self._event_queue = event_queue
        self._stop_event = stop_event
        self._model_family = model_family
        self._model_size = model_size
        self._conf_threshold = conf_threshold
        self._device = device
        self._max_frames = max_frames
        self._use_tensorrt = use_tensorrt
        self._pipeline_kwargs = pipeline_kwargs

    def run(self):
        from src.video import VideoLoader
        from src.detection import YOLODetector
        from src.tracking.tracker import Tracker
        from src.analytics.zones import ZoneManager
        from src.analytics.events import EventDetector
        from src.analytics.abandoned import AbandonedDetector
        from src.models.event import EventStore

        loader = VideoLoader(self._video_path)
        detector = YOLODetector(
            model_family=self._model_family,
            model_size=self._model_size,
            device=self._device,
            use_tensorrt=self._use_tensorrt,
        )

        def send_event(event_type: str, track_id: int, data: dict):
            self._event_queue.put({
                "camera_id": self._cam_id,
                "event_type": event_type,
                "track_id": track_id,
                "timestamp": time.time(),
                **data,
            })

        events = EventStore()
        total_frames = (
            min(loader.frame_count, self._max_frames)
            if self._max_frames else loader.frame_count
        )
        pbar = tqdm(
            total=total_frames,
            desc=f"Camera {self._cam_id}",
            position=self._cam_id,
        )

        for i, frame in enumerate(loader):
            if self._stop_event.is_set():
                break
            if self._max_frames and i >= self._max_frames:
                break

            detections = detector.detect(frame, conf_threshold=self._conf_threshold)
            pbar.update(1)

        pbar.close()
        loader.release()
        self._event_queue.put({
            "camera_id": self._cam_id,
            "event_type": "_done",
            "track_id": -1,
            "timestamp": time.time(),
            "total_frames": total_frames,
        })


class MultiCameraPipeline:
    def __init__(self, camera_configs: list[dict], output_dir: str = "outputs"):
        self._camera_configs = camera_configs
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._workers: list[CameraWorker] = []
        self._queue = Queue()
        self._stop_event = MP_Event()

    def run(self) -> dict:
        original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        self._workers = []
        for i, cfg in enumerate(self._camera_configs):
            w = CameraWorker(
                camera_id=i,
                video_path=cfg["video_path"],
                output_dir=str(self._output_dir / f"camera_{i}"),
                event_queue=self._queue,
                stop_event=self._stop_event,
                model_family=cfg.get("model_family", "yolo11"),
                model_size=cfg.get("model_size", "nano"),
                conf_threshold=cfg.get("conf_threshold", 0.4),
                device=cfg.get("device", "cuda"),
                max_frames=cfg.get("max_frames"),
                use_tensorrt=cfg.get("use_tensorrt", False),
            )
            self._workers.append(w)

        signal.signal(signal.SIGINT, original_sigint)

        for w in self._workers:
            w.start()

        collected_events = []
        completed = set()
        total_workers = len(self._workers)

        try:
            with tqdm(total=total_workers, desc="Cameras") as pbar:
                while len(completed) < total_workers:
                    msg = self._queue.get()
                    if msg.get("event_type") == "_done":
                        cam_id = msg["camera_id"]
                        if cam_id not in completed:
                            completed.add(cam_id)
                            pbar.update(1)
                    else:
                        collected_events.append(msg)
        except KeyboardInterrupt:
            print("\nStopping all cameras...")
        finally:
            self._stop_event.set()
            for w in self._workers:
                w.join(timeout=5)
                if w.is_alive():
                    w.kill()

        result = {
            "total_cameras": total_workers,
            "completed_cameras": len(completed),
            "events": collected_events,
            "event_count": len(collected_events),
        }
        report_path = self._output_dir / "multi_camera_report.json"
        report_path.write_text(json.dumps(result, indent=2))
        print(f"\nReport: {report_path}")
        return result
