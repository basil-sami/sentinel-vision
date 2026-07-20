import json
import logging
import signal
import sys
import time
import multiprocessing as mp

# Use spawn so each child gets its own CUDA context (fork breaks CUDA)
_ctx = mp.get_context("spawn")
Process = _ctx.Process
Queue = _ctx.Queue
Event = _ctx.Event
from pathlib import Path


class CameraWorker(Process):
    def __init__(
        self,
        camera_id: int,
        video_path: str,
        output_dir: str,
        event_queue: Queue,
        stop_event: Event,
        **pipeline_kwargs,
    ):
        super().__init__()
        self._cam_id = camera_id
        self._video_path = video_path
        self._output_dir = output_dir
        self._event_queue = event_queue
        self._stop_event = stop_event
        self._pipeline_kwargs = pipeline_kwargs

    VALID_PARAMS = {
        "model_family", "model_size", "conf_threshold", "device",
        "max_frames", "track_thresh", "match_thresh", "track_low_thresh",
        "track_buffer", "trail_length", "use_reid", "reid_model",
        "zone_config", "calibration_config", "capture_evidence",
        "filter_stationary_objects", "min_move_distance",
        "target_classes", "use_tensorrt", "log_level",
    }

    def run(self):
        from src.pipeline import analyze_video
        from src.logging_setup import setup_logging

        log = setup_logging(
            f"cam{self._cam_id}",
            log_dir=self._output_dir,
            level=logging.DEBUG,
            console=False,
        )
        log.info("=== CameraWorker %d start ===", self._cam_id)
        log.info("video=%s  output=%s", self._video_path, self._output_dir)
        log.info("kwargs=%s", {k: v for k, v in self._pipeline_kwargs.items()
                                if k in self.VALID_PARAMS})

        try:
            kwargs = {k: v for k, v in self._pipeline_kwargs.items()
                      if k in self.VALID_PARAMS}
            kwargs["log_level"] = logging.DEBUG
            result = analyze_video(
                video_path=self._video_path,
                output_dir=self._output_dir,
                **kwargs,
            )
            log.info("Result: %d tracks, %d events, %d frames",
                     result["total_objects_tracked"],
                     len(result["events"]),
                     result.get("total_frames_processed", 0))
            payload = {
                "tracks": result["total_objects_tracked"],
                "detections": result["total_detections"],
                "events": len(result["events"]),
                "frames": result.get("total_frames_processed", 0),
                "object_counts": result.get("object_counts", {}),
                "vehicles": result.get("vehicles", {}),
                "scene_events": result.get("scene_events", {}),
                "gate_counts": result.get("gate_counts", {}),
                "output_video": result.get("output_video", ""),
                "error": None,
            }
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log.error("CameraWorker %d FAILED: %s", self._cam_id, e)
            for line in tb.strip().split("\n")[-10:]:
                log.error("  %s", line)
            payload = {
                "tracks": 0, "detections": 0, "events": 0, "frames": 0,
                "object_counts": {},
                "vehicles": {}, "scene_events": {}, "gate_counts": {},
                "output_video": "",
                "error": f"{type(e).__name__}: {e}",
                "traceback": tb,
            }

        self._event_queue.put({
            "camera_id": self._cam_id,
            "event_type": "_done",
            "track_id": -1,
            "timestamp": time.time(),
            "result": payload,
        })


class MultiCameraPipeline:
    def __init__(self, camera_configs: list[dict], output_dir: str = "outputs"):
        self._camera_configs = camera_configs
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._workers: list[CameraWorker] = []
        self._queue = Queue()
        self._stop_event = Event()

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
                        if cam_id in completed:
                            continue
                        completed.add(cam_id)
                        res = msg.get("result", {})
                        err = res.get("error")
                        if err:
                            print(f"\n  ⚠ camera_{cam_id} ERROR: {err}")
                            tb = res.get("traceback", "")
                            if tb:
                                for line in tb.strip().split("\n")[-3:]:
                                    print(f"    {line}")
                        results[f"camera_{cam_id}"] = res
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
