#!/usr/bin/env python3
"""Split detect() into GPU compute vs CPU post-processing via CUDA events.

Measures per frame:
  - wall_ms:    Python wall clock for model.predict()
  - gpu_ms:     GPU kernel execution via CUDA events
  - cpu_ms:     wall_ms - gpu_ms (preprocess + NMS + box decode + Python)
  - parse_ms:   iterating result.boxes to build Detection list
  - overhead:   cpu_ms - parse_ms (NMS inside ultralytics)

Run:
  python scripts/bench_detect_split.py --video "The CCTV People Demo 2.mp4" --frames 200
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="The CCTV People Demo 2.mp4")
    parser.add_argument("--frames", type=int, default=200)
    args = parser.parse_args()

    frames = load_frames(args.video, args.frames)
    print(f"Loaded {len(frames)} frames")

    detector = YOLODetector(
        model_family="yolo11",
        model_size="xlarge",
        device="cuda",
        use_tensorrt=True,
    )

    # Warmup
    print("Warmup...")
    for f in frames[:10]:
        detector.detect(f, conf_threshold=0.4)
    torch.cuda.synchronize()
    print("Done.")

    metrics = {
        "wall_ms": [],
        "gpu_ms": [],
        "cpu_ms": [],
        "parse_ms": [],
        "overhead_ms": [],
    }

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for i, f in enumerate(frames):
        def parse(results):
            t0 = time.perf_counter()
            dets = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    if class_id not in detector.target_classes:
                        continue
                    _ = map(int, box.xyxy[0].tolist())
                    _ = float(box.conf[0])
            return time.perf_counter() - t0

        # --- Single predict() call with both wall and GPU timing ---
        start_event.record()
        t_wall0 = time.perf_counter()

        results = detector.model.predict(
            f, conf=0.4, device="cuda", verbose=False,
        )

        end_event.record()
        torch.cuda.synchronize()
        t_wall1 = time.perf_counter()

        gpu_ms = start_event.elapsed_time(end_event)
        wall_ms = (t_wall1 - t_wall0) * 1000
        cpu_ms = wall_ms - gpu_ms

        parse_s = parse(results)
        parse_ms = parse_s * 1000
        overhead_ms = cpu_ms - parse_ms

        metrics["wall_ms"].append(wall_ms)
        metrics["gpu_ms"].append(gpu_ms)
        metrics["cpu_ms"].append(cpu_ms)
        metrics["parse_ms"].append(parse_ms)
        metrics["overhead_ms"].append(overhead_ms)

        if (i + 1) % 50 == 0:
            print(f"  Frame {i+1}/{len(frames)}")

    # Report
    print(f"\n{'─' * 60}")
    print(f"  detect() Stage Timing ({len(frames)} frames)")
    print(f"{'─' * 60}")
    print(f"  {'Stage':<30s} {'Avg (ms)':>10s} {'Min (ms)':>10s} {'Max (ms)':>10s}")
    print(f"  {'─' * 60}")
    for stage in ["wall_ms", "gpu_ms", "cpu_ms", "parse_ms", "overhead_ms"]:
        vals = metrics[stage]
        label = {
            "wall_ms": "Total detect()",
            "gpu_ms": "GPU compute (CUDA events)",
            "cpu_ms": "CPU (wall - gpu)",
            "parse_ms": "Box parsing (Python)",
            "overhead_ms": "Preproc + NMS (ultralytics)",
        }[stage]
        print(f"  {label:<30s} {sum(vals)/len(vals):>10.2f} {min(vals):>10.2f} {max(vals):>10.2f}")

    print(f"\n  GPU share:     {sum(metrics['gpu_ms'])/sum(metrics['wall_ms'])*100:.1f}% of detect()")
    print(f"  CPU share:     {sum(metrics['cpu_ms'])/sum(metrics['wall_ms'])*100:.1f}% of detect()")
    print(f"  Parse share:   {sum(metrics['parse_ms'])/sum(metrics['gpu_ms'] + metrics['cpu_ms'])*100:.1f}% of (GPU+CPU)")
    print(f"  Effective FPS: {len(frames) / (sum(metrics['wall_ms'])/1000):.1f}")

    # Visual breakdown per frame (every 10th)
    print(f"\n  Per-frame breakdown (every 10th frame):")
    print(f"  {'Frame':>6s} {'Wall':>6s} {'GPU':>6s} {'CPU':>6s} {'Parse':>6s} {'─' * 20}")
    for i in range(0, len(frames), 10):
        w = metrics["wall_ms"][i]
        g = metrics["gpu_ms"][i]
        c = metrics["cpu_ms"][i]
        p = metrics["parse_ms"][i]
        bar_g = "█" * max(1, int(g / 2))
        bar_c = "░" * max(1, int(c / 2))
        print(f"  {i:>6d} {w:>6.1f} {g:>6.1f} {c:>6.1f} {p:>6.1f}  {bar_g}{bar_c}")


if __name__ == "__main__":
    main()
