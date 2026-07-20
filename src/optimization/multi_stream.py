import json
import signal
import sys
import time
from multiprocessing import Process, Queue, Event as MP_Event
from pathlib import Path


class CameraWorker(Process):
    def __init__(
        self,
        camera_id: int,
        video_path: str,
        output_dir: str,
        event_queue: Queue,
        stop_event: MP_Event,
        **pipeline_kwargs,
    ):
        super().__init__()
        self._cam_id = camera_id
        self._video_path = video_path
        self._output_dir = output_dir
        self._event_queue = event_queue
        self._stop_event = stop_event
        self._pipeline_kwargs = pipeline_kwargs

    def run(self):
        from src.pipeline import analyze_video

        result = analyze_video(
            video_path=self._video_path,
            output_dir=self._output_dir,
            **self._pipeline_kwargs,
        )

        self._event_queue.put({
            "camera_id": self._cam_id,
            "event_type": "_done",
            "track_id": -1,
            "timestamp": time.time(),
            "result": {
                "tracks": result["total_objects_tracked"],
                "detections": result["total_detections"],
                "events": len(result["events"]),
                "object_counts": result.get("object_counts", {}),
                "vehicles": result.get("vehicles", {}),
                "scene_events": result.get("scene_events", {}),
                "gate_counts": result.get("gate_counts", {}),
                "output_video": result.get("output_video", ""),
            },
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
            pipe_kwargs = {k: v for k, v in cfg.items()
                           if k not in ("video_path", "output_dir")}
            w = CameraWorker(
                camera_id=i,
                video_path=cfg["video_path"],
                output_dir=str(self._output_dir / f"camera_{i}"),
                event_queue=self._queue,
                stop_event=self._stop_event,
                **pipe_kwargs,
            )
            self._workers.append(w)

        signal.signal(signal.SIGINT, original_sigint)

        for w in self._workers:
            w.start()

        results = {}
        completed = set()
        total_workers = len(self._workers)

        try:
            from tqdm import tqdm
            with tqdm(total=total_workers, desc="Cameras") as pbar:
                while len(completed) < total_workers:
                    msg = self._queue.get()
                    if msg.get("event_type") == "_done":
                        cam_id = msg["camera_id"]
                        if cam_id not in completed:
                            completed.add(cam_id)
                            results[f"camera_{cam_id}"] = msg.get("result", {})
                            pbar.update(1)
        except KeyboardInterrupt:
            print("\nStopping all cameras...")
        finally:
            self._stop_event.set()
            for w in self._workers:
                w.join(timeout=10)
                if w.is_alive():
                    w.kill()

        report = {
            "total_cameras": total_workers,
            "completed_cameras": len(completed),
            "per_camera": results,
            "event_count": sum(
                r.get("events", 0) for r in results.values()
            ),
        }
        report_path = self._output_dir / "multi_camera_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"\nReport: {report_path}")
        return report
