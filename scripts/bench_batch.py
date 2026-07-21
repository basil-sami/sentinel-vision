#!/usr/bin/env python3
"""Benchmark dynamic-batch TensorRT engines across batch sizes 1, 2, 4, 8.

Exports engines with batch sizes as needed, then measures:
  - Throughput (FPS) at each batch size
  - GPU utilization (nvidia-smi)
  - GPU memory usage
  - Latency per batch and per frame

Usage:
  python scripts/bench_batch.py --video "The CCTV People Demo 2.mp4" --frames 200
"""

import argparse
import subprocess
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimization.tensorrt_export import export_to_engine


BATCH_SIZES = [1, 2, 4, 8]


def load_frames(video_path: str, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"  Loaded {len(frames)} frames")
    return frames


def sample_gpu(stop_event, results_list, interval=0.2):
    samples = []
    while not stop_event.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            parts = out.stdout.strip().split(",")
            if len(parts) >= 2:
                samples.append({
                    "util": float(parts[0].strip()),
                    "mem_used": float(parts[1].strip()),
                    "mem_total": float(parts[2].strip()) if len(parts) >= 3 else 0,
                })
        except Exception:
            pass
        time.sleep(interval)
    results_list.extend(samples)


def benchmark_batch(detector, frames: list[np.ndarray], batch_size: int,
                    warmup: int = 10) -> dict:
    """Run N frames through detect_batch() in batch_size chunks."""
    # Build batches
    batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
    # Drop incomplete last batch
    if len(batches[-1]) < batch_size:
        batches = batches[:-1]

    if not batches:
        return {"fps": 0, "latency_ms": 0, "total_frames": 0}

    # Warmup
    for b in batches[:warmup // batch_size + 1]:
        detector.detect_batch(b, conf_threshold=0.4)

    # GPU sampling
    stop_event = threading.Event()
    gpu_samples = []
    sampler = threading.Thread(target=sample_gpu, args=(stop_event, gpu_samples, 0.2))
    sampler.start()

    # Timed run
    latencies = []
    t0 = time.perf_counter()
    for b in batches:
        t1 = time.perf_counter()
        detector.detect_batch(b, conf_threshold=0.4)
        latencies.append((time.perf_counter() - t1) * 1000)
    elapsed = time.perf_counter() - t0
    torch.cuda.synchronize()

    stop_event.set()
    sampler.join()

    total_frames = sum(len(b) for b in batches)
    fps = total_frames / elapsed if elapsed > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    result = {
        "batch_size": batch_size,
        "batches": len(batches),
        "total_frames": total_frames,
        "elapsed": elapsed,
        "fps": fps,
        "avg_latency_ms": avg_latency,
        "per_frame_ms": avg_latency / batch_size,
        "gpu_samples": gpu_samples,
    }

    return result


def print_result(batch_size: int, r: dict):
    utils = [s["util"] for s in r["gpu_samples"]] if r["gpu_samples"] else []
    mems = [s["mem_used"] for s in r["gpu_samples"]] if r["gpu_samples"] else []

    print(f"  batch={batch_size}")
    print(f"    Frames:           {r['total_frames']}")
    print(f"    Elapsed:          {r['elapsed']:.2f}s")
    print(f"    Throughput:       {r['fps']:.0f} FPS")
    print(f"    Per-batch:        {r['avg_latency_ms']:.1f} ms")
    print(f"    Per-frame:        {r['per_frame_ms']:.2f} ms")
    if utils:
        print(f"    GPU util (avg):   {sum(utils)/len(utils):.1f}%")
        print(f"    GPU util (peak):  {max(utils):.1f}%")
    if mems:
        print(f"    GPU mem (avg):    {sum(mems)/len(mems):.0f} MiB")
        print(f"    GPU mem (peak):   {max(mems):.0f} MiB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="The CCTV People Demo 2.mp4")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=BATCH_SIZES)
    parser.add_argument("--force-export", action="store_true",
                        help="Re-export engines even if they exist")
    parser.add_argument("--model", default="xlarge",
                        choices=["medium", "large", "xlarge"])
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print(f"  Batch Size Benchmark — YOLO11{args.model} TensorRT")
    print(f"{'=' * 60}")

    frames = load_frames(args.video, args.frames)

    results = {}

    for bs in sorted(args.batch_sizes):
        print(f"\n{'─' * 50}")
        print(f"  Exporting engine (batch={bs})...")

        engine = export_to_engine(
            model_family="yolo11",
            model_size=args.model,
            half=True,
            device=0,
            force=args.force_export,
            batch_size=bs,
        )
        print(f"  Engine: {engine}")

        # Load detector pointing at this engine
        # We load directly via YOLO to bypass YOLODetector's batch=1 assumption
        from ultralytics import YOLO
        model = YOLO(engine)
        print(f"  Loaded. Warming up...")

        # Warmup
        warmup_frames = frames[:10]
        warmup_batches = [warmup_frames[i:i + bs]
                          for i in range(0, len(warmup_frames), bs)]
        for b in warmup_batches:
            if len(b) == bs:
                model.predict(b, conf=0.4, device="cuda", verbose=False)
        torch.cuda.synchronize()

        # Build a wrapper so we can use predict() directly
        class DetectorWrapper:
            def __init__(self, m):
                self.model = m
            def detect_batch(self, imgs, conf_threshold=0.4):
                return self.model.predict(imgs, conf=conf_threshold,
                                          device="cuda", verbose=False)

        detector = DetectorWrapper(model)
        print(f"  Running benchmark...")

        r = benchmark_batch(detector, frames, bs)
        results[bs] = r
        print_result(bs, r)

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Batch':>6s} {'FPS':>8s} {'ms/batch':>10s} {'ms/frame':>10s} {'GPU util':>10s} {'GPU mem':>10s}")
    print(f"  {'─' * 54}")
    for bs in sorted(results.keys()):
        r = results[bs]
        utils = [s["util"] for s in r["gpu_samples"]] if r["gpu_samples"] else []
        mems = [s["mem_used"] for s in r["gpu_samples"]] if r["gpu_samples"] else []
        util_avg = f"{sum(utils)/len(utils):.0f}%" if utils else "?"
        mem_avg = f"{sum(mems)/len(mems):.0f}" if mems else "?"
        print(f"  {bs:>6d} {r['fps']:>8.0f} {r['avg_latency_ms']:>10.1f} {r['per_frame_ms']:>10.2f} {util_avg:>10s} {mem_avg:>10s}")

    # Efficiency analysis
    if 1 in results:
        base_fps = results[1]["fps"]
        print(f"\n  Scaling efficiency (vs batch=1 baseline):")
        for bs in sorted(results.keys()):
            if bs == 1:
                continue
            r = results[bs]
            ideal = base_fps * bs
            actual = r["fps"]
            eff = actual / ideal * 100
            print(f"    batch={bs}: {actual:.0f} FPS vs ideal {ideal:.0f} FPS = {eff:.0f}% efficient")


if __name__ == "__main__":
    main()
