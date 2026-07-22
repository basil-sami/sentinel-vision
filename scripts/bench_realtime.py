#!/usr/bin/env python3
"""Real-time camera stream simulator — measures analysis lag.

Simulates live camera feeds and measures how far behind real-time
the pipeline falls.

Usage:
  # Single camera, real-time (1x speed)
  python scripts/bench_realtime.py --video "The CCTV People Demo 2.mp4"

  # Single camera, 2x stress test
  python scripts/bench_realtime.py --video "..." --speed 2.0

  # 4 cameras, real-time
  python scripts/bench_realtime.py --video "..." --multi

  # 4 cameras, 4x stress
  python scripts/bench_realtime.py --video "..." --multi --speed 4.0
"""

import argparse
import csv
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection import YOLODetector
from src.tracking.tracker import Tracker
from src.analytics.identity import IdentityConfidence
from src.analytics.prediction import TrackPredictor
from src.analytics.correlation import EventCorrelator
from src.analytics.time_sync import TimeSync
from src.analytics.face_recognition import FaceRecognizer
from src.analytics.vehicle.orchestrator import VehicleAnalyzer


# ── helpers ──────────────────────────────────────────────────────────

def _load_frames(video_path: str, max_frames: int = 0) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps


def _generate_feeds(source_video: str, test_dir: str = "test_videos") -> list[tuple[str, str]]:
    td = Path(test_dir)
    td.mkdir(exist_ok=True)
    transforms = [
        ("cam01_mirror_slow", "hflip,setpts=2.0*PTS"),
        ("cam02_mirror_reverse", "hflip,reverse"),
        ("cam03_slow_reverse", "reverse,setpts=2.0*PTS"),
    ]
    feeds = [("cam00_original", source_video)]
    for name, filt in transforms:
        path = td / f"{name}.mp4"
        if not path.exists():
            print(f"  Generating {name}...")
            subprocess.run(
                ["ffmpeg", "-y", "-i", source_video, "-vf", filt,
                 "-an", "-c:v", "libx264", "-preset", "fast", str(path)],
                capture_output=True,
            )
        feeds.append((name, str(path)))
    return feeds


# ── per-camera capture + pipeline thread ──────────────────────────────

class CameraSimulator:
    """Simulates a live camera feed and runs the full analysis pipeline."""

    def __init__(self, name: str, frames: list[np.ndarray],
                 source_fps: float, speed: float,
                 detector: YOLODetector, conf_threshold: float = 0.4):
        self.name = name
        self.frames = frames
        self.source_fps = source_fps
        self.speed = speed
        self.detector = detector
        self.conf_threshold = conf_threshold
        self.frame_interval = 1.0 / (source_fps * speed)

        # Pipeline stages
        self.tracker = Tracker(
            track_thresh=0.4, track_low_thresh=0.1, track_buffer=450,
            match_thresh=0.7, use_reid=False, device=detector.device,
            use_cmc=False,
        )
        self.identity = IdentityConfidence()
        self.predictor = TrackPredictor()
        self.correlator = EventCorrelator()
        self.time_sync = TimeSync(fps=source_fps)
        self.vehicle = VehicleAnalyzer(plate_read_interval=10)
        self.face_recognizer = FaceRecognizer(device=detector.device)
        self._face_interval = max(1, int(source_fps / 5))  # ~5 FPS face check

        # Results
        self.lag_records: list[dict] = []
        self.stage_times: list[dict] = []
        self.dropped = 0
        self.processed = 0
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        t_start = time.perf_counter()
        for i, frame in enumerate(self.frames):
            if self._stop.is_set():
                break

            expected_capture_time = t_start + i * self.frame_interval
            now = time.perf_counter()
            if now - expected_capture_time > 1.0:
                self.dropped += 1
                continue

            sleep = expected_capture_time - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)

            capture_t = time.perf_counter()

            # ── Pipeline stages ──
            stages = {}

            t0 = time.perf_counter()
            detections = self.detector.detect(frame, conf_threshold=self.conf_threshold)
            stages["detect"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            tracks = self.tracker.update(detections, frame, frame_index=i)
            stages["track"] = time.perf_counter() - t0

            for t in tracks:
                self.identity.update(t.id, t.bbox, t.confidence, i)
                cx = (t.bbox[0] + t.bbox[2]) // 2
                cy = (t.bbox[1] + t.bbox[3]) // 2
                self.predictor.update(t.id, cx, cy, i)

            t0 = time.perf_counter()
            self.vehicle.process_frame(frame, tracks, i)
            stages["vehicle"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            face_events = self.face_recognizer.process_frame(frame, tracks, i) if self.face_recognizer.available else []
            stages["face"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            ts = self.time_sync.frame_timestamp(i)
            stages["timesync"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            inc_events = []
            for ev in face_events:
                inc = self.correlator.process_event({
                    "type": ev.get("type", "face_recognized"),
                    "track_id": ev.get("track_id", -1),
                    "timestamp": ts.utc_timestamp,
                    "severity": "medium",
                })
                if inc:
                    inc_events.append(inc)
            stages["correlate"] = time.perf_counter() - t0

            pipe_end = time.perf_counter()

            lag = pipe_end - expected_capture_time
            self.lag_records.append({
                "frame": i,
                "capture_t": capture_t,
                "pipe_end": pipe_end,
                "lag_s": round(lag, 4),
                "detections": len(detections),
                "tracks": len(tracks),
            })
            self.stage_times.append(stages)
            self.processed += 1

    def summary(self) -> dict:
        if not self.lag_records:
            return {"processed": 0, "dropped": self.dropped, "avg_lag": 0, "max_lag": 0, "p95_lag": 0, "p99_lag": 0}
        lags = [r["lag_s"] for r in self.lag_records]
        sorted_lags = sorted(lags)
        return {
            "processed": self.processed,
            "dropped": self.dropped,
            "avg_lag": round(sum(lags) / len(lags), 3),
            "max_lag": round(max(lags), 3),
            "p95_lag": round(sorted_lags[int(len(sorted_lags) * 0.95)], 3),
            "p99_lag": round(sorted_lags[int(len(sorted_lags) * 0.99)], 3),
        }


# ── report ────────────────────────────────────────────────────────────

def _print_report(results: list[CameraSimulator], output_csv: str | None):
    print(f"\n{'=' * 70}")
    print(f"  REAL-TIME LAG REPORT")
    print(f"{'=' * 70}")

    # Per camera
    all_lags = []
    for sim in results:
        s = sim.summary()
        print(f"\n  {sim.name}:")
        print(f"    Processed:   {s['processed']} frames")
        print(f"    Dropped:     {s['dropped']} frames")
        print(f"    Avg lag:     {s['avg_lag']*1000:.0f} ms")
        print(f"    Max lag:     {s['max_lag']*1000:.0f} ms")
        print(f"    P95 lag:     {s['p95_lag']*1000:.0f} ms")
        print(f"    P99 lag:     {s['p99_lag']*1000:.0f} ms")
        all_lags.extend(r["lag_s"] for r in sim.lag_records)

    # System summary
    if all_lags:
        sorted_lags = sorted(all_lags)
        total_frames = sum(sim.processed for sim in results)
        total_dropped = sum(sim.dropped for sim in results)
        print(f"\n  {'─' * 50}")
        print(f"  SYSTEM TOTAL ({len(results)} camera{'s' if len(results) > 1 else ''}):")
        print(f"    Frames processed:  {total_frames}")
        print(f"    Frames dropped:    {total_dropped}")
        print(f"    Avg lag:           {sum(all_lags)/len(all_lags)*1000:.0f} ms")
        print(f"    Max lag:           {max(all_lags)*1000:.0f} ms")
        print(f"    P95 lag:           {sorted_lags[int(len(sorted_lags)*0.95)]*1000:.0f} ms")
        print(f"    P99 lag:           {sorted_lags[int(len(sorted_lags)*0.99)]*1000:.0f} ms")
        # Categorize
        healthy = sum(1 for l in all_lags if l < 0.1)
        lagging = sum(1 for l in all_lags if 0.1 <= l < 0.5)
        critical = sum(1 for l in all_lags if l >= 0.5)
        print(f"    Lag distribution:")
        print(f"      Healthy (<100ms):  {healthy} frames ({healthy/len(all_lags)*100:.0f}%)")
        print(f"      Lagging (100-500ms): {lagging} frames ({lagging/len(all_lags)*100:.0f}%)")
        print(f"      Critical (>500ms):   {critical} frames ({critical/len(all_lags)*100:.0f}%)")

    # CSV
    # Stage timing breakdown
    all_stages = {}
    for sim in results:
        for rec in sim.stage_times:
            for stage, t in rec.items():
                all_stages.setdefault(stage, []).append(t)
    if all_stages:
        print(f"\n  Per-frame stage times (avg across all cameras):")
        for stage in ["detect", "track", "vehicle", "face", "timesync", "correlate"]:
            if stage in all_stages:
                vals = all_stages[stage]
                print(f"    {stage:<12s} {sum(vals)/len(vals)*1000:>6.1f} ms avg  ({min(vals)*1000:.1f}-{max(vals)*1000:.1f} ms)")

    if output_csv:
        with open(output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["camera", "frame", "capture_t", "pipe_end", "lag_s", "detections", "tracks",
                         "detect_ms", "track_ms", "vehicle_ms", "face_ms", "timesync_ms", "correlate_ms"])
            for sim in results:
                for j, r in enumerate(sim.lag_records):
                    st = sim.stage_times[j] if j < len(sim.stage_times) else {}
                    w.writerow([sim.name, r["frame"], r["capture_t"], r["pipe_end"],
                                r["lag_s"], r["detections"], r["tracks"],
                                round(st.get("detect", 0)*1000, 2),
                                round(st.get("track", 0)*1000, 2),
                                round(st.get("vehicle", 0)*1000, 2),
                                round(st.get("face", 0)*1000, 2),
                                round(st.get("timesync", 0)*1000, 2),
                                round(st.get("correlate", 0)*1000, 2)])
        print(f"\n  CSV: {output_csv}")


# ── main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-time camera simulator")
    parser.add_argument("--video", default="The CCTV People Demo 2.mp4")
    parser.add_argument("--frames", type=int, default=500,
                        help="Frames to process per camera")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (1.0 = real-time)")
    parser.add_argument("--multi", action="store_true",
                        help="Simulate 4 cameras instead of 1")
    parser.add_argument("--model-size", default="medium",
                        choices=["nano", "small", "medium", "large", "xlarge"])
    parser.add_argument("--tensorrt", action="store_true")
    parser.add_argument("--output-csv", default=None,
                        help="Save per-frame lag data to CSV")
    args = parser.parse_args()

    print(f"{'=' * 70}")
    print(f"  REAL-TIME CAMERA SIMULATOR")
    print(f"  Model: YOLO11{args.model_size} {'TensorRT' if args.tensorrt else 'PyTorch'}")
    print(f"  Speed: {args.speed}x  |  Frames: {args.frames}  |  Cameras: {'4' if args.multi else '1'}")
    print(f"{'=' * 70}")

    # Load detector once (shared across cameras)
    print("\nLoading detector...")
    detector = YOLODetector(
        model_family="yolo11",
        model_size=args.model_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_tensorrt=args.tensorrt,
    )
    print("  Done.")

    # Prepare feeds
    if args.multi:
        feeds = _generate_feeds(args.video)
    else:
        feeds = [("camera", args.video)]

    # Load frames for each feed
    cameras: list[CameraSimulator] = []
    for name, path in feeds:
        frames, fps = _load_frames(path, args.frames)
        if not frames:
            print(f"  WARNING: no frames from {path}, skipping")
            continue
        print(f"\n  {name}: {len(frames)} frames @ {fps:.0f} FPS (simulating {fps*args.speed:.0f} FPS)")
        cameras.append(CameraSimulator(name, frames, fps, args.speed, detector))

    if not cameras:
        print("No cameras to simulate.")
        return

    # Run simulations
    threads = []
    t_wall0 = time.perf_counter()
    for cam in cameras:
        t = threading.Thread(target=cam.run, daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    wall_elapsed = time.perf_counter() - t_wall0

    # Report
    _print_report(cameras, args.output_csv)

    # Timing breakdown
    total_frames = sum(cam.processed for cam in cameras)
    print(f"\n  Wall elapsed: {wall_elapsed:.1f}s")
    print(f"  Effective system FPS: {total_frames / wall_elapsed:.0f}")

    # Compare with offline FPS (detector-only benchmark)
    offline_fps = {"nano": 200, "small": 150, "medium": 110, "large": 95, "xlarge": 52}
    fps_ref = offline_fps.get(args.model_size, 110)
    if fps_ref:
        fps_cam = total_frames / wall_elapsed
        print(f"  Offline FPS (YOLO-only): ~{fps_ref}")
        print(f"  Real-time headroom:      {fps_ref / (fps_cam / len(cameras) + 0.01):.1f}x")


if __name__ == "__main__":
    main()
