#!/usr/bin/env python3
"""Concurrent YOLO inference: 4 processes sharing 1 GPU.

Simulates batching without rebuilding the TRT engine.

Measures:
  - Throughput per process and total
  - GPU utilization (nvidia-smi from parent)
  - Per-process latency distribution

Run:
  python scripts/bench_concurrent.py --video "The CCTV People Demo 2.mp4" --frames 200
"""

import argparse
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


WORKERS = 4


def load_frames(video_path: str, n_per_worker: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < n_per_worker:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def worker(worker_id: int, frames: list[np.ndarray], result_queue, warmup: int = 10):
    """Each worker loads its own detector and runs inference."""
    import torch
    from src.detection import YOLODetector

    detector = YOLODetector(
        model_family="yolo11",
        model_size="xlarge",
        device="cuda",
        use_tensorrt=True,
    )

    # Warmup
    for f in frames[:warmup]:
        detector.detect(f, conf_threshold=0.4)
    torch.cuda.synchronize()

    # Timed run
    latencies = []
    t0 = time.perf_counter()
    for f in frames:
        t1 = time.perf_counter()
        dets = detector.detect(f, conf_threshold=0.4)
        latencies.append((time.perf_counter() - t1) * 1000)
    elapsed = time.perf_counter() - t0
    torch.cuda.synchronize()

    result_queue.put({
        "worker": worker_id,
        "frames": len(frames),
        "elapsed": elapsed,
        "fps": len(frames) / elapsed,
        "total_dets": sum(len(d) for d in [detector.detect(f, conf_threshold=0.4) for f in frames[:0]]),  # skip
        "latencies": latencies,
    })


def sample_gpu(duration: float, interval: float = 0.2, result_queue=None):
    """Sample GPU utilization in a background process."""
    samples = []
    t_end = time.time() + duration + 2.0
    while time.time() < t_end:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            line = out.stdout.strip()
            parts = line.split(",")
            if len(parts) >= 2:
                samples.append({
                    "util": float(parts[0].strip()),
                    "mem_used": float(parts[1].strip()),
                    "mem_total": float(parts[2].strip()) if len(parts) >= 3 else 0,
                })
        except Exception:
            pass
        time.sleep(interval)
    if result_queue:
        result_queue.put(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="The CCTV People Demo 2.mp4")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()

    frames_per_worker = args.frames
    total_frames = frames_per_worker * args.workers

    # Load frames once and split among workers
    all_frames = load_frames(args.video, total_frames)
    print(f"Loaded {len(all_frames)} frames ({args.workers} workers × {frames_per_worker})")

    # Try to split, but if we don't have enough frames, duplicate
    if len(all_frames) < total_frames:
        # Duplicate to fill
        while len(all_frames) < total_frames:
            all_frames.extend(all_frames[:total_frames - len(all_frames)])
    worker_frames = [all_frames[i * frames_per_worker:(i + 1) * frames_per_worker]
                     for i in range(args.workers)]

    result_queue = multiprocessing.Queue()
    gpu_queue = multiprocessing.Queue()

    # Estimate runtime for GPU sampling
    est_runtime = frames_per_worker / 50.0 * 1.5  # ~50 fps per worker, 50% safety

    gpu_proc = multiprocessing.Process(
        target=sample_gpu, args=(est_runtime, 0.2, gpu_queue)
    )
    gpu_proc.start()

    procs = []
    for i in range(args.workers):
        p = multiprocessing.Process(
            target=worker, args=(i, worker_frames[i], result_queue)
        )
        procs.append(p)

    t0 = time.perf_counter()
    for p in procs:
        p.start()

    for p in procs:
        p.join()
    total_elapsed = time.perf_counter() - t0

    gpu_proc.join(timeout=3)

    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    gpu_samples = []
    while not gpu_queue.empty():
        gpu_samples.extend(gpu_queue.get())

    # Report
    print(f"\n{'=' * 60}")
    print(f"  Concurrent YOLO Inference ({args.workers} workers × {frames_per_worker} frames)")
    print(f"{'=' * 60}")

    total_fps = 0
    all_latencies = []
    for r in sorted(results, key=lambda x: x["worker"]):
        total_fps += r["fps"]
        all_latencies.extend(r["latencies"])
        print(f"\n  Worker {r['worker']}:")
        print(f"    Frames:        {r['frames']}")
        print(f"    Elapsed:       {r['elapsed']:.2f}s")
        print(f"    Throughput:    {r['fps']:.1f} FPS")
        print(f"    Avg latency:   {sum(r['latencies'])/len(r['latencies']):.1f} ms")

    print(f"\n  {'─' * 50}")
    print(f"  System total:")
    print(f"    Wall time:     {total_elapsed:.2f}s")
    print(f"    Total frames:  {args.workers * frames_per_worker}")
    print(f"    System FPS:    {args.workers * frames_per_worker / total_elapsed:.1f}")
    print(f"    Combined FPS:  {total_fps:.1f} (sum of workers)")
    print(f"    GPU sharing overhead: {(1 - (total_fps / (53.7 * args.workers))) * 100:.1f}% vs ideal")
    print(f"    Avg latency:   {sum(all_latencies)/len(all_latencies):.1f} ms")
    print(f"    P50 latency:   {sorted(all_latencies)[len(all_latencies)//2]:.1f} ms")
    print(f"    P95 latency:   {sorted(all_latencies)[int(len(all_latencies)*0.95)]:.1f} ms")

    if gpu_samples:
        utils = [s["util"] for s in gpu_samples]
        mems = [s["mem_used"] for s in gpu_samples]
        print(f"\n    GPU util (avg): {sum(utils)/len(utils):.1f}%")
        print(f"    GPU util (peak): {max(utils):.1f}%")
        print(f"    GPU mem (avg):   {sum(mems)/len(mems):.0f} MiB")
        print(f"    GPU mem (peak):  {max(mems):.0f} MiB")

        # Compare with single-process baseline
        print(f"\n  vs Single-process baseline:")
        print(f"    Single FPS:    53.7 (from Test 1)")
        print(f"    {args.workers}-process FPS: {args.workers * frames_per_worker / total_elapsed:.1f}")
        print(f"    Scaling:       {(args.workers * frames_per_worker / total_elapsed) / 53.7:.2f}x")
        print(f"    GPU util jump: {sum(utils)/len(utils) - 35.0:+.1f} percentage points")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
