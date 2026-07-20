"""Multi-camera pipeline — thread pool (parallel CPU, shared GPU)."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def process_cameras(
    camera_configs: list[dict],
    output_dir: str = "outputs",
    mosaic_layout: str = "2x2",
    max_workers: int = 4,
) -> dict:
    """Process N cameras in parallel using a thread pool.

    A single shared YOLO detector is used across all camera threads.
    Each camera runs its own tracker/zone/event state independently.
    GPU inference is serialized by CUDA (safe for concurrent calls).
    CPU post-processing runs in parallel across threads.
    """
    from src.pipeline import analyze_video
    from src.detection import YOLODetector

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create one shared detector from the first config's parameters
    if camera_configs:
        first = camera_configs[0]
        shared_detector = YOLODetector.shared(
            model_family=first.get("model_family", "yolo11"),
            model_size=first.get("model_size", "xlarge"),
            device=first.get("device", "cuda"),
            target_classes=first.get("target_classes"),
            use_tensorrt=first.get("use_tensorrt", False),
        )
    else:
        shared_detector = YOLODetector.shared()

    def _run_one(i: int, cfg: dict) -> tuple[str, dict]:
        cam_key = f"camera_{i}"
        cam_out = str(output_dir / cam_key)
        pipe_kwargs = {k: v for k, v in cfg.items()
                       if k not in ("video_path", "output_dir", "name")}

        print(f"\n{'='*50}")
        print(f"Camera {i}: {cfg.get('name', cam_key)}")
        print(f"  Video: {cfg['video_path']}")
        print(f"  Output: {cam_out}")
        print(f"{'='*50}")

        result = analyze_video(
            video_path=cfg["video_path"],
            output_dir=cam_out,
            detector=shared_detector,
            **pipe_kwargs,
        )

        summary = {
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
        print(f"  ✓ {summary['tracks']} tracks, {summary['events']} events, "
              f"{summary['frames']} frames")
        return cam_key, summary

    results: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one, i, cfg): i
                   for i, cfg in enumerate(camera_configs)}

        for f in as_completed(futures):
            i = futures[f]
            try:
                cam_key, summary = f.result()
                results[cam_key] = summary
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                cam_key = f"camera_{i}"
                print(f"\n  ✗ Camera {i} FAILED: {e}")
                for line in tb.strip().split("\n")[-5:]:
                    print(f"    {line}")
                results[cam_key] = {
                    "tracks": 0, "detections": 0, "events": 0, "frames": 0,
                    "object_counts": {}, "vehicles": {}, "scene_events": {},
                    "gate_counts": {}, "output_video": "",
                    "error": f"{type(e).__name__}: {e}",
                }

    report = {
        "total_cameras": len(camera_configs),
        "completed_cameras": len(results),
        "per_camera": results,
        "event_count": sum(r.get("events", 0) for r in results.values()),
    }
    report_path = output_dir / "multi_camera_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {report_path}")

    # Auto-generate mosaic from all successfully processed cameras
    output_videos = {}
    for cam_key, cam_res in results.items():
        vid = cam_res.get("output_video", "")
        if vid and Path(vid).exists() and cam_res.get("error") is None:
            idx = int(cam_key.split("_")[-1])
            cfg = camera_configs[idx] if 0 <= idx < len(camera_configs) else {}
            label = cfg.get("name", cam_key)
            output_videos[label] = vid

    if len(output_videos) >= 2:
        try:
            from src.visualization.mosaic import create_mosaic
            mosaic_path = str(output_dir / "mosaic.mp4")
            create_mosaic(output_videos, mosaic_path, layout=mosaic_layout)
            report["mosaic"] = mosaic_path
        except Exception as e:
            print(f"  Mosaic generation skipped: {e}")

    return report


class MultiCameraPipeline:
    def __init__(self, camera_configs, output_dir="outputs",
                 mosaic_layout="2x2", max_workers=4):
        self._camera_configs = camera_configs
        self._output_dir = output_dir
        self._mosaic_layout = mosaic_layout
        self._max_workers = max_workers

    def run(self) -> dict:
        return process_cameras(
            self._camera_configs,
            self._output_dir,
            mosaic_layout=self._mosaic_layout,
            max_workers=self._max_workers,
        )
