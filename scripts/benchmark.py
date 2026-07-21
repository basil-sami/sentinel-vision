#!/usr/bin/env python3
"""Controlled experiments to determine why GPU is idle.

Runs 7 tests:

  Test 1 — GPU Stress: YOLO-only loop, no CPU pipeline
  Test 2 — Disable Vehicle: pipeline without vehicle intelligence
  Test 3 — Overlap: CPU decode vs GPU inference interleaved
  Test 4 — Batch Scale: YOLO batch=1,2,4,8 throughput
  Test 5 — Worker Scale: CPU workers=1,2,4,8,16
  Test 6 — Idle Gaps: microsecond-level GPU busy/idle timeline
  Test 7 — Transfer Cost: Host↔Device copy latency
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection import YOLODetector
from src.optimization.profiler import PipelineProfiler


def load_test_frames(video_path: str, n: int = 100) -> list[np.ndarray]:
    """Load N frames from a video file into a list of RGB arrays."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"  Loaded {len(frames)} frames from {video_path}")
    return frames


def print_header(name: str):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")


def print_result(label: str, value: str):
    print(f"  {label:<40s} {value}")


# ──────────────────────────────────────────────
# Test 1 — GPU Stress Test
# ──────────────────────────────────────────────
def test_gpu_stress(detector: YOLODetector, frames: list[np.ndarray], warmup: int = 20):
    print_header("Test 1 — GPU Stress Test (YOLO-only, no CPU pipeline)")

    profiler = PipelineProfiler()
    profiler.start_system_sampling(interval=0.2)

    # Warmup
    for f in frames[:warmup]:
        detector.detect(f, conf_threshold=0.4)

    # Timed run
    t0 = time.perf_counter()
    total_dets = 0
    for f in frames:
        dets = detector.detect(f, conf_threshold=0.4)
        total_dets += len(dets)
        profiler.record_frame_time((time.perf_counter() - t0) * 1000 / len(frames))
    elapsed = time.perf_counter() - t0

    profiler.stop_system_sampling()
    fps = len(frames) / elapsed

    print_result("Frames processed", f"{len(frames)}")
    print_result("Elapsed", f"{elapsed:.2f}s")
    print_result("YOLO-only FPS", f"{fps:.1f}")
    print_result("Total detections", f"{total_dets}")
    print_result("Avg dets/frame", f"{total_dets / len(frames):.1f}")

    if profiler._gpu_samples:
        utils = [s["util"] for s in profiler._gpu_samples]
        print_result("GPU util (avg)", f"{sum(utils)/len(utils):.1f}%")
        print_result("GPU util (peak)", f"{max(utils):.1f}%")

    return fps


# ──────────────────────────────────────────────
# Test 2 — Remove Vehicle Intelligence
# ──────────────────────────────────────────────
def test_no_vehicle(video_path: str, detector: YOLODetector, max_frames: int = 200):
    print_header("Test 2 — Pipeline Without Vehicle Intelligence")

    from src.pipeline import analyze_video

    t0 = time.perf_counter()
    result = analyze_video(
        video_path=video_path,
        output_dir="bench_no_vehicle",
        model_family="yolo11",
        model_size="xlarge",
        conf_threshold=0.4,
        device="cuda",
        max_frames=max_frames,
        use_tensorrt=True,
        use_cmc=False,
        plate_read_interval=10,
        detector=detector,
        capture_evidence=False,
        use_reid=True,
        reid_refresh_interval=50,
        reid_new_track_frames=3,
    )
    elapsed = time.perf_counter() - t0
    fps = max_frames / elapsed

    print_result("Frames", f"{max_frames}")
    print_result("Elapsed", f"{elapsed:.2f}s")
    print_result("Pipeline FPS", f"{fps:.1f}")
    print_result("Tracks", f"{result['total_objects_tracked']}")
    print_result("Events", f"{len(result['events'])}")

    # Compare with full pipeline (reference from earlier run)
    # Full pipeline was 79ms/frame → 12.7 FPS
    print_result("vs full pipeline (ref)", "12.7 FPS")
    print_result("Speedup", f"{fps / 12.7:.1f}x")

    return fps


# ──────────────────────────────────────────────
# Test 3 — CPU/GPU Overlap
# ──────────────────────────────────────────────
def test_overlap(detector: YOLODetector, frames: list[np.ndarray]):
    print_header("Test 3 — CPU/GPU Overlap Simulation")

    import threading

    gpu_timestamps = []
    cpu_timestamps = []
    lock = threading.Lock()
    done = threading.Event()

    def gpu_worker():
        for f in frames:
            with lock:
                gpu_timestamps.append(time.perf_counter())
            detector.detect(f, conf_threshold=0.4)
            with lock:
                gpu_timestamps.append(time.perf_counter())
        done.set()

    def cpu_worker():
        while not done.is_set():
            with lock:
                cpu_timestamps.append(time.perf_counter())
            time.sleep(0.005)  # simulate CPU work

    gpu_thread = threading.Thread(target=gpu_worker, daemon=True)
    cpu_thread = threading.Thread(target=cpu_worker, daemon=True)

    t0_prof = PipelineProfiler()
    t0_prof.start_system_sampling(interval=0.1)

    t0 = time.perf_counter()
    gpu_thread.start()
    cpu_thread.start()
    gpu_thread.join()
    done.set()
    cpu_thread.join()
    elapsed = time.perf_counter() - t0

    t0_prof.stop_system_sampling()

    fps = len(frames) / elapsed

    # Calculate GPU busy ratio from timestamps
    gpu_busy = 0.0
    for i in range(0, len(gpu_timestamps) - 1, 2):
        gpu_busy += gpu_timestamps[i + 1] - gpu_timestamps[i]
    total = gpu_timestamps[-1] - gpu_timestamps[0] if len(gpu_timestamps) >= 2 else 1
    gpu_util_pct = gpu_busy / total * 100 if total > 0 else 0

    print_result("Frames", f"{len(frames)}")
    print_result("Elapsed", f"{elapsed:.2f}s")
    print_result("FPS (overlapped)", f"{fps:.1f}")
    print_result("GPU busy ratio", f"{gpu_util_pct:.1f}%")
    print_result("CPU/GPU overlap", "Yes" if gpu_util_pct > 40 else "No (serialized)")

    if t0_prof._gpu_samples:
        utils = [s["util"] for s in t0_prof._gpu_samples]
        print_result("GPU util (avg)", f"{sum(utils)/len(utils):.1f}%")

    return fps, gpu_util_pct


# ──────────────────────────────────────────────
# Test 4 — Batch Size Scaling
# ──────────────────────────────────────────────
def test_batch_scale(detector: YOLODetector, frames: list[np.ndarray]):
    print_header("Test 4 — Batch Size Experiment")

    dups = [frames[0] for _ in range(64)]

    for batch_size in [1, 2, 4]:
        batches = [dups[i:i + batch_size] for i in range(0, len(dups), batch_size)]

        # Warmup
        try:
            for b in batches[:4]:
                if len(b) == 1:
                    detector.detect(b[0], conf_threshold=0.4)
                else:
                    detector.detect_batch(b, conf_threshold=0.4)
        except Exception as e:
            reason = str(e).split('\n')[0][:80]
            print(f"\n  batch={batch_size} — SKIPPED: {reason}")
            continue

        t0 = time.perf_counter()
        total_frames = 0
        for b in batches:
            if len(b) == 1:
                detector.detect(b[0], conf_threshold=0.4)
            else:
                detector.detect_batch(b, conf_threshold=0.4)
            total_frames += len(b)
        elapsed = time.perf_counter() - t0

        fps = total_frames / elapsed
        ms_per_batch = elapsed / len(batches) * 1000
        latency = ms_per_batch / batch_size

        print(f"\n  batch={batch_size}")
        print_result("  Total frames", f"{total_frames}")
        print_result("  Elapsed", f"{elapsed:.2f}s")
        print_result("  Throughput", f"{fps:.0f} fps")
        print_result("  Per-batch", f"{ms_per_batch:.1f} ms")
        print_result("  Per-frame latency", f"{latency:.2f} ms")


# ──────────────────────────────────────────────
# Test 5 — CPU Worker Scaling
# ──────────────────────────────────────────────
def test_worker_scale(video_path: str, detector: YOLODetector, max_frames: int = 200):
    print_header("Test 5 — CPU Worker Scaling (via process_cameras)")

    from src.optimization.multi_stream import process_cameras

    # Create 4 identical camera configs
    base_cfg = {
        "video_path": video_path,
        "model_size": "xlarge", "conf_threshold": 0.4,
        "device": "cuda", "use_tensorrt": True,
        "use_cmc": False, "plate_read_interval": 10,
        "reid_refresh_interval": 50, "reid_new_track_frames": 3,
        "capture_evidence": False, "filter_stationary_objects": True,
        "min_move_distance": 20.0, "max_frames": max_frames,
    }
    cams = [base_cfg.copy() for _ in range(4)]

    for workers in [1, 2, 4]:
        t0 = time.perf_counter()
        report = process_cameras(cams, f"bench_workers_{workers}",
                                 max_workers=workers)
        elapsed = time.perf_counter() - t0

        total_frames = sum(r.get("frames", 0) for r in report["per_camera"].values())
        total_events = sum(r.get("events", 0) for r in report["per_camera"].values())
        system_fps = total_frames / elapsed

        print(f"\n  workers={workers}")
        print_result("  4 cams total time", f"{elapsed:.0f}s")
        print_result("  System throughput", f"{system_fps:.1f} fps")
        print_result("  Per-camera FPS", f"{system_fps/4:.1f}")
        print_result("  Total events", f"{total_events}")


# ──────────────────────────────────────────────
# Test 6 — GPU Idle Gaps
# ──────────────────────────────────────────────
def test_idle_gaps(detector: YOLODetector, frames: list[np.ndarray]):
    print_header("Test 6 — GPU Idle Gap Measurement")

    timestamps = []
    for i, f in enumerate(frames[:100]):
        t_start = time.perf_counter()
        dets = detector.detect(f, conf_threshold=0.4)
        t_end = time.perf_counter()
        timestamps.append((t_start, t_end, len(dets)))

    gaps = []
    for i in range(1, len(timestamps)):
        gap_start = timestamps[i - 1][1]
        gap_end = timestamps[i][0]
        gaps.append((gap_end - gap_start) * 1000)

    gpu_times = [(t[1] - t[0]) * 1000 for t in timestamps]
    gpu_busy = sum(gpu_times)
    total_time = (timestamps[-1][1] - timestamps[0][0]) * 1000
    gpu_pct = gpu_busy / total_time * 100 if total_time > 0 else 0

    print_result("GPU busy time", f"{gpu_busy:.0f} ms")
    print_result("Total wall time", f"{total_time:.0f} ms")
    print_result("GPU busy ratio", f"{gpu_pct:.1f}%")
    print_result("GPU idle ratio", f"{100 - gpu_pct:.1f}%")
    print_result("Avg GPU burst", f"{sum(gpu_times)/len(gpu_times):.1f} ms")
    print_result("Avg idle gap", f"{sum(gaps)/len(gaps):.1f} ms" if gaps else "N/A")
    print_result("Max idle gap", f"{max(gaps):.1f} ms" if gaps else "N/A")

    # Timeline (sample every 10 frames)
    print("\n  GPU Timeline (every 10th frame):")
    for i in range(0, len(timestamps), 10):
        bar_len = int(gpu_times[i] / 2)
        gap_bar = 0
        if i > 0:
            gap_bar = int(gaps[i - 1] / 2)
        bar = "█" * min(bar_len, 40)
        gap_str = "·" * min(gap_bar, 20)
        print(f"  Frame {i:>3d}: GPU {gap_str}{bar}  ({gpu_times[i]:.0f}ms)")

    return gpu_pct


# ──────────────────────────────────────────────
# Test 7 — Transfer Cost
# ──────────────────────────────────────────────
def test_transfer(detector: YOLODetector, frames: list[np.ndarray]):
    print_header("Test 7 — CPU/GPU Transfer Cost")

    # Host-to-device: torch.from_numpy + cuda()
    sample = frames[0]
    times_h2d = []
    for _ in range(50):
        t0 = time.perf_counter()
        t = torch.from_numpy(sample).cuda()
        torch.cuda.synchronize()
        times_h2d.append((time.perf_counter() - t0) * 1000)
        del t

    # Device-to-host: .cpu().numpy()
    t_gpu = torch.from_numpy(sample).cuda()
    times_d2h = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = t_gpu.cpu().numpy()
        torch.cuda.synchronize()
        times_d2h.append((time.perf_counter() - t0) * 1000)

    # Full inference tensor round-trip (what actually happens per frame)
    times_full = []
    for f in frames[:50]:
        t0 = time.perf_counter()
        dets = detector.detect(f, conf_threshold=0.4)
        times_full.append((time.perf_counter() - t0) * 1000)

    print_result("Host → Device (640×480 RGB)", f"{sum(times_h2d)/len(times_h2d):.2f} ms")
    print_result("Device → Host (640×480 RGB)", f"{sum(times_d2h)/len(times_d2h):.2f} ms")
    print_result("Full inference round-trip", f"{sum(times_full)/len(times_full):.1f} ms")
    print_result("Transfer as % of inference",
                 f"{(sum(times_h2d)/len(times_h2d) + sum(times_d2h)/len(times_d2h)) / (sum(times_full)/len(times_full)) * 100:.1f}%")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GPU/CPU Bottleneck Benchmarks")
    parser.add_argument("--video", default="The CCTV People Demo 2.mp4")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--test", type=int, nargs="+",
                        choices=range(1, 8),
                        help="Run specific tests only (e.g. --test 1 4 6)")
    args = parser.parse_args()

    print("=" * 60)
    print("  SENTINEL VISION — GPU/CPU BOTTLENECK BENCHMARK")
    print("=" * 60)

    # Load test frames once
    frames = load_test_frames(args.video, args.frames)
    if not frames:
        print("ERROR: no frames loaded")
        sys.exit(1)

    # Create shared detector
    print("\nLoading YOLO detector (TensorRT)...")
    detector = YOLODetector(
        model_family="yolo11",
        model_size="xlarge",
        device="cuda",
        use_tensorrt=True,
    )
    print("  Done.")

    tests_to_run = set(args.test) if args.test else {1, 2, 3, 4, 6, 7}

    results = {}

    if 1 in tests_to_run:
        results["gpu_stress"] = test_gpu_stress(detector, frames)

    if 2 in tests_to_run:
        results["no_vehicle"] = test_no_vehicle(args.video, detector, args.frames)

    if 3 in tests_to_run:
        results["overlap"] = test_overlap(detector, frames)

    if 4 in tests_to_run:
        test_batch_scale(detector, frames)

    if 5 in tests_to_run:
        test_worker_scale(args.video, detector, args.frames)

    if 6 in tests_to_run:
        results["idle_gaps"] = test_idle_gaps(detector, frames)

    if 7 in tests_to_run:
        test_transfer(detector, frames)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    gpu_stress = results.get("gpu_stress", 0)
    no_vehicle = results.get("no_vehicle", 0)
    overlap = results.get("overlap", (0, 0))

    if gpu_stress:
        print_result("Test 1 — GPU stress FPS", f"{gpu_stress:.0f}")
    if no_vehicle:
        print_result("Test 2 — No-vehicle FPS", f"{no_vehicle:.0f}")
    if overlap:
        print_result("Test 3 — Overlap GPU util", f"{overlap[1]:.1f}%")
        print_result("Test 3 — Overlap FPS", f"{overlap[0]:.0f}")


if __name__ == "__main__":
    main()
