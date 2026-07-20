"""Multi-camera pipeline — sequential (single process, shared GPU)."""

import json
import logging
import time
from pathlib import Path


def process_cameras(
    camera_configs: list[dict],
    output_dir: str = "outputs",
) -> dict:
    """Process N cameras sequentially in a single process (shared GPU context).

    Each camera gets independent tracker/zone/event state but uses the same
    detector model.  This avoids all multiprocessing+CUDA deadlocks.
    """
    from src.pipeline import analyze_video
    from tqdm import tqdm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for i, cfg in enumerate(camera_configs):
        cam_key = f"camera_{i}"
        cam_out = str(output_dir / cam_key)
        pipe_kwargs = {k: v for k, v in cfg.items()
                       if k not in ("video_path", "output_dir", "name")}

        print(f"\n{'='*50}")
        print(f"Camera {i}: {cfg.get('name', cam_key)}")
        print(f"  Video: {cfg['video_path']}")
        print(f"  Output: {cam_out}")
        print(f"{'='*50}")

        try:
            result = analyze_video(
                video_path=cfg["video_path"],
                output_dir=cam_out,
                **pipe_kwargs,
            )
            results[cam_key] = {
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
            r = results[cam_key]
            print(f"  ✓ {r['tracks']} tracks, {r['events']} events, "
                  f"{r['frames']} frames")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"  ✗ FAILED: {e}")
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
    return report


# Kept for backward compatibility
class MultiCameraPipeline:
    def __init__(self, camera_configs, output_dir="outputs"):
        self._camera_configs = camera_configs
        self._output_dir = output_dir

    def run(self) -> dict:
        return process_cameras(self._camera_configs, self._output_dir)
