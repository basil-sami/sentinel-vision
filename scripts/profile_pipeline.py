#!/usr/bin/env python3
"""Performance profiler for the analysis pipeline.

Measures per-stage timing, GPU/CPU utilization, call counters, and
generates a formatted investigation report.

Usage:
    python scripts/profile_pipeline.py                          # single camera defaults
    python scripts/profile_pipeline.py --video <path> --frames 500
    python scripts/profile_pipeline.py --multi                   # 4-feed test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimization.profiler import PipelineProfiler
from src.video import VideoLoader
from src.detection import YOLODetector
from src.tracking.tracker import Tracker
from src.analytics.object_history import ObjectHistory
from src.analytics.zones import ZoneManager
from src.analytics.counting import GateCounter
from src.analytics.dwell import DwellTracker
from src.analytics.events import EventDetector
from src.analytics.abandoned import AbandonedDetector
from src.analytics.calibration import Calibrator
from src.analytics.interaction import InteractionModel
from src.analytics.evidence import EvidenceCapture
from src.analytics.vehicle.orchestrator import VehicleAnalyzer
from src.analytics.scene.orchestrator import SceneAnalyzer
from src.models.event import EventStore, Event
from src.visualization import Annotator
from src.visualization.zone_renderer import draw_zones, draw_gates, draw_event_ticker


def profile_video(
    video_path: str,
    output_dir: str = "profile_output",
    model_family: str = "yolo11",
    model_size: str = "xlarge",
    conf_threshold: float = 0.4,
    device: str = "cuda",
    max_frames: int | None = 500,
    use_tensorrt: bool = False,
    use_cmc: bool = False,
    plate_read_interval: int = 10,
    reid_refresh_interval: int = 50,
    reid_new_track_frames: int = 3,
    trail_length: int = 50,
    track_buffer: int = 450,
) -> PipelineProfiler:
    """Run video analysis with full per-stage profiling."""
    profiler = PipelineProfiler()
    profiler.start_system_sampling(interval=0.2)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Init components
    loader = VideoLoader(video_path)
    total = min(loader.frame_count, max_frames) if max_frames else loader.frame_count

    detector = YOLODetector(
        model_family=model_family,
        model_size=model_size,
        device=device,
        use_tensorrt=use_tensorrt,
    )

    tracker = Tracker(
        track_thresh=0.4,
        track_buffer=track_buffer,
        use_reid=True,
        reid_model="x1_0",
        device=device,
        use_cmc=use_cmc,
        reid_refresh_interval=reid_refresh_interval,
        reid_new_track_frames=reid_new_track_frames,
    )

    history = ObjectHistory()
    events = EventStore()
    zone_mgr = ZoneManager()
    calibrator = Calibrator()
    gate_counter = GateCounter()
    dwell_tracker = DwellTracker()
    event_detector = EventDetector()
    abandoned_detector = AbandonedDetector(stationary_threshold_frames=track_buffer)
    interaction_model = InteractionModel()
    vehicle_analyzer = VehicleAnalyzer(plate_read_interval=plate_read_interval)
    scene_analyzer = SceneAnalyzer()

    annotator = Annotator(
        output_path=str(output_dir / "output_tracking.mp4"),
        fps=loader.fps,
        width=loader.width,
        height=loader.height,
    )

    evidence = EvidenceCapture(
        str(output_dir),
        fps=loader.fps,
        width=loader.width,
        height=loader.height,
    )

    _zone_state: dict[int, set[str]] = {}

    print(f"Profiling {total} frames of {video_path}...")
    frame_start = time.perf_counter()

    for i, frame in enumerate(loader):
        if max_frames and i >= max_frames:
            break

        # --- Decode is implicit (frame reading above) ---

        # --- YOLO Detection ---
        with profiler.timer("detect"):
            detections = detector.detect(frame, conf_threshold=conf_threshold)
        profiler.count("detections", len(detections))

        # --- Tracking ---
        with profiler.timer("track"):
            tracks = tracker.update(detections, frame, frame_index=i)
        profiler.count("tracks", len(tracks))

        # --- History ---
        with profiler.timer("history"):
            history.update(tracks, i)

        # --- Preprocessing ---
        with profiler.timer("preprocess"):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if evidence:
                evidence.add_frame(frame_bgr)

        # --- Zone & Gate Logic ---
        with profiler.timer("zones"):
            for t in tracks:
                cx = (t.bbox[0] + t.bbox[2]) // 2
                cy = (t.bbox[1] + t.bbox[3]) // 2

                current_zones = set(z.name for z in zone_mgr.zones_at(cx, cy))
                prev_zones = _zone_state.get(t.id, set())

                for zn in current_zones - prev_zones:
                    ev = event_detector.check_zone_entry(t.id, t.class_name, zn, cx, cy)
                    events.add(ev)
                for zn in prev_zones - current_zones:
                    dwell_s = dwell_tracker.current_dwell(t.id, zn, i, loader.fps)
                    ev = event_detector.check_zone_exit(t.id, t.class_name, zn, dwell_s, cx, cy)
                    events.add(ev)

                for zn in current_zones:
                    dwell_tracker.update(t.id, zn, i, True)
                    dwell_s = dwell_tracker.current_dwell(t.id, zn, i, loader.fps)
                    loiter_ev = event_detector.check_loitering(t.id, t.class_name, zn, dwell_s, i, cx, cy)
                    if loiter_ev:
                        events.add(loiter_ev)
                for zn in prev_zones - current_zones:
                    dwell_tracker.update(t.id, zn, i, False)

                _zone_state[t.id] = current_zones

                for gate_name, direction in zone_mgr.check_gate_crossing(t.id, cx, cy):
                    gate_counter.record(gate_name, t.id, direction)

                if t.class_name != "person":
                    ab_ev = abandoned_detector.update(t.id, t.class_name, t.bbox, i, tracks)
                    if ab_ev:
                        events.add(ab_ev)

        # --- Interaction ---
        with profiler.timer("interaction"):
            interaction_events = interaction_model.update(tracks, i)
            for iev in interaction_events:
                events.add(iev)

        # --- Vehicle Intelligence ---
        with profiler.timer("vehicle"):
            vehicle_events = vehicle_analyzer.process_frame(frame, tracks, i, calibrator)
            for ve in vehicle_events:
                events.add(ve)

        # --- Scene Understanding ---
        with profiler.timer("scene"):
            scene_events = scene_analyzer.process_frame(tracks, i, calibrator, zone_mgr)
            for se in scene_events:
                events.add(se)

        # --- Render ---
        with profiler.timer("render"):
            annotated = annotator.draw_tracks(frame, tracks, history, trail_length=trail_length)
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            annotated_bgr = draw_event_ticker(annotated_bgr, events.all())
            annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        # --- Encode ---
        with profiler.timer("encode"):
            annotator.write_frame(annotated)

        frame_ms = (time.perf_counter() - frame_start) * 1000
        profiler.record_frame_time(frame_ms)
        frame_start = time.perf_counter()

        avg = profiler.avg_frame_ms
        remaining = ((total - i - 1) * avg) / 1000 if avg else 0
        sys.stdout.write(
            f"\r  Frame {i+1}/{total}  |  "
            f"detect {profiler.stages['detect'].avg_ms:.0f}ms  "
            f"track {profiler.stages['track'].avg_ms:.0f}ms  "
            f"vehicle {profiler.stages['vehicle'].avg_ms:.0f}ms  "
            f"render {profiler.stages['render'].avg_ms:.0f}ms  "
            f"encode {profiler.stages['encode'].avg_ms:.0f}ms  "
            f"| avg {avg:.0f}ms/fr  ETA {remaining:.0f}s  "
        )
        sys.stdout.flush()

    print()
    annotator.release()
    loader.release()
    profiler.stop_system_sampling()

    # ReID call count from tracker internals
    profiler.count("reid_calls", tracker._track_ages.__len__())  # rough proxy

    return profiler


def profile_multi(camera_configs, output_dir="profile_multi", max_frames=500):
    """Profile multi-camera sequential processing."""
    from src.optimization.multi_stream import process_cameras

    profiler = PipelineProfiler()
    profiler.start_system_sampling(interval=0.2)

    t0 = time.perf_counter()
    report = process_cameras(camera_configs, output_dir)
    elapsed = time.perf_counter() - t0

    profiler.stop_system_sampling()

    print(f"\nMulti-camera: {len(camera_configs)} cams, {elapsed:.0f}s total")
    for ck, cr in report.get("per_camera", {}).items():
        fr = cr.get("frames", 0)
        fps = fr / elapsed if elapsed else 0
        profiler.count(f"{ck}_frames", fr)
        profiler.count(f"{ck}_tracks", cr.get("tracks", 0))
        profiler.count(f"{ck}_events", cr.get("events", 0))
        print(f"  {ck}: {fr} frames, {fps:.1f} fps, {cr.get('tracks')} tracks")

    return profiler


def main():
    parser = argparse.ArgumentParser(description="Pipeline Performance Profiler")
    parser.add_argument("--video", default="The CCTV People Demo 2.mp4")
    parser.add_argument("--output", default="profile_output")
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--model", default="xlarge")
    parser.add_argument("--tensorrt", action="store_true")
    parser.add_argument("--multi", action="store_true", help="Profile 4-feed multi-camera")
    args = parser.parse_args()

    if args.multi:
        # Generate test videos if they don't exist
        import subprocess
        test_dir = Path("test_videos")
        test_dir.mkdir(exist_ok=True)
        transforms = [
            ("cam01_mirror_slow", "hflip,setpts=2.0*PTS"),
            ("cam02_mirror_reverse", "hflip,reverse"),
            ("cam03_slow_reverse", "reverse,setpts=2.0*PTS"),
        ]
        feeds = []

        def _ensure_video(name: str, filt: str) -> str:
            p = test_dir / f"{name}.mp4"
            if not p.exists():
                print(f"Generating {name}...")
                subprocess.run(["ffmpeg", "-y", "-i", args.video, "-vf", filt,
                               "-an", "-c:v", "libx264", "-preset", "fast", str(p)],
                               capture_output=True)
            return str(p)

        feeds = [("cam00_original", args.video)]
        for nm, filt in transforms:
            feeds.append((nm, _ensure_video(nm, filt)))

        cam_configs = [
            {"video_path": vp, "name": nm,
             "model_size": args.model, "conf_threshold": 0.4,
             "device": "cuda", "use_tensorrt": args.tensorrt,
             "use_cmc": False, "plate_read_interval": 10,
             "reid_refresh_interval": 50, "reid_new_track_frames": 3,
             "capture_evidence": True, "filter_stationary_objects": True,
             "min_move_distance": 20.0, "max_frames": args.frames}
            for nm, vp in feeds
        ]

        profiler = profile_multi(cam_configs, args.output + "_multi", args.frames)
    else:
        profiler = profile_video(
            video_path=args.video,
            output_dir=args.output,
            model_size=args.model,
            use_tensorrt=args.tensorrt,
            max_frames=args.frames,
        )

    print()
    print(profiler.report())

    report_path = Path(args.output) / "performance_report.txt"
    report_path.write_text(profiler.report())
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
