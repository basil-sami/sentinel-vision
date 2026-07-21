#!/usr/bin/env python3
"""Benchmark YOLO11 model sizes (medium, large, xlarge) on TensorRT batch=1.

Measures:
  - Throughput (FPS)
  - GPU utilization
  - GPU memory
  - Latency

Usage:
  python scripts/bench_models.py --video "The CCTV People Demo 2.mp4" --frames 200
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


SIZES = ["medium", "large", "xlarge"]
FLOPS = {"medium": 72, "large": 107, "xlarge": 195}
MAP = {"medium": 72.3, "large": 74.0, "xlarge": 75.3}


def load_frames(video_path: str, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
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


def benchmark_model(model, frames, warmup=20):
    # Warmup
    for f in frames[:warmup]:
        model.predict(f, conf=0.4, device="cuda", verbose=False)
    torch.cuda.synchronize()

    stop_event = threading.Event()
    gpu_samples = []
    sampler = threading.Thread(target=sample_gpu,
                               args=(stop_event, gpu_samples, 0.2))
    sampler.start()

    latencies = []
    t0 = time.perf_counter()
    for f in frames:
        t1 = time.perf_counter()
        model.predict(f, conf=0.4, device="cuda", verbose=False)
        latencies.append((time.perf_counter() - t1) * 1000)
    elapsed = time.perf_counter() - t0
    torch.cuda.synchronize()

    stop_event.set()
    sampler.join()

    fps = len(frames) / elapsed
    return {
        "fps": fps,
        "elapsed": elapsed,
        "avg_latency": sum(latencies) / len(latencies),
        "gpu_samples": gpu_samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="The CCTV People Demo 2.mp4")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--sizes", nargs="+", default=SIZES,
                        choices=SIZES)
    parser.add_argument("--force-export", action="store_true")
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print(f"  Model Size Benchmark — TensorRT batch=1")
    print(f"{'=' * 60}")

    frames = load_frames(args.video, args.frames)
    print(f"  Loaded {len(frames)} frames\n")

    results = {}
    for size in args.sizes:
        print(f"  {'─' * 50}")
        print(f"  Model: YOLO11{size} ({FLOPS[size]} GFLOPs, mAP50={MAP[size]})")
        print(f"  Exporting engine...")

        engine = export_to_engine(
            model_family="yolo11",
            model_size=size,
            half=True,
            device=0,
            force=args.force_export,
            batch_size=1,
        )
        print(f"  Engine: {engine}")

        from ultralytics import YOLO
        model = YOLO(engine)

        print(f"  Benchmarking...")
        r = benchmark_model(model, frames)
        results[size] = r

        utils = [s["util"] for s in r["gpu_samples"]] if r["gpu_samples"] else []
        mems = [s["mem_used"] for s in r["gpu_samples"]] if r["gpu_samples"] else []

        print(f"    FPS:              {r['fps']:.0f}")
        print(f"    Elapsed:          {r['elapsed']:.2f}s")
        print(f"    Avg latency:      {r['avg_latency']:.1f} ms")
        if utils:
            print(f"    GPU util (avg):   {sum(utils)/len(utils):.1f}%")
            print(f"    GPU util (peak):  {max(utils):.1f}%")
        if mems:
            print(f"    GPU mem (avg):    {sum(mems)/len(mems):.0f} MiB")
            print(f"    GPU mem (peak):   {max(mems):.0f} MiB")

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY — Model Size Comparison")
    print(f"{'=' * 60}")
    print(f"  {'Model':>10s} {'GFLOPs':>8s} {'FPS':>8s} {'Latency':>10s} {'GPU util':>10s} {'GPU mem':>10s}")
    print(f"  {'─' * 56}")
    for size in args.sizes:
        r = results[size]
        utils = [s["util"] for s in r["gpu_samples"]] if r["gpu_samples"] else []
        mems = [s["mem_used"] for s in r["gpu_samples"]] if r["gpu_samples"] else []
        util_avg = f"{sum(utils)/len(utils):.0f}%" if utils else "?"
        mem_avg = f"{sum(mems)/len(mems):.0f}" if mems else "?"
        print(f"  {'YOLO11'+size:>10s} {FLOPS[size]:>8d} {r['fps']:>8.0f} "
              f"{r['avg_latency']:>9.1f}ms {util_avg:>10s} {mem_avg:>10s}")


if __name__ == "__main__":
    main()
