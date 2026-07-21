# Performance Analysis

## Profiler Results (Single Camera, TensorRT YOLO11x, T4)

500 frames of "The CCTV People Demo 2.mp4" (640×360, 25fps).

### Per-Frame Timing

| Stage          | Avg (ms) | Min (ms) | Max (ms) | % Runtime | Calls |
|----------------|----------|----------|----------|-----------|-------|
| detect         | 28.0     | 15.0     | 2720.8   | 36%       | 500   |
| vehicle        | 27.6     | 0.0      | 3888.6   | 35%       | 500   |
| encode         | 13.5     | 3.3      | 51.1     | 17%       | 500   |
| track          | 5.8      | 0.4      | 20.2     | 7%        | 500   |
| render         | 1.9      | 0.3      | 17.6     | 3%        | 500   |
| interaction    | 0.4      | 0.0      | 3.0      | <1%       | 500   |
| zones          | 0.4      | 0.0      | 2.8      | <1%       | 500   |
| preprocess     | 0.1      | 0.1      | 3.8      | <1%       | 500   |
| scene          | 0.1      | 0.0      | 0.4      | <1%       | 500   |
| history        | 0.0      | 0.0      | 3.1      | <1%       | 500   |

**Total: 78.9 ms/frame → 12.7 FPS**

### System Utilization

| Metric         | Value              | Verdict                    |
|----------------|--------------------|----------------------------|
| GPU util (avg) | 17.9%              | **GPU idle 82% of time**   |
| GPU util (peak)| 54.0%              | Never fully loaded          |
| GPU memory     | 366 / 15360 MiB   | **Barely using it**         |
| CPU util (avg) | 75.9%              | CPU saturated               |
| CPU util (peak)| 100.0%             | Hitting the ceiling         |
| ReID calls     | 13 (across 500 fr) | Caching works (was ~4000)   |

### Bottleneck Ranking (by total time)

| Rank | Component   | Total (ms) | %    | CPU/GPU | Expected Speedup |
|------|-------------|------------|------|---------|-----------------|
| 1    | detect      | 14015      | 36%  | GPU     | Medium (batch)   |
| 2    | vehicle     | 13793      | 35%  | GPU/CPU | Low (one-time)   |
| 3    | **encode**  | **6758**   | **17%** | **I/O** | **High**         |
| 4    | track       | 2876       | 7%   | CPU     | Low              |
| 5    | render      | 956        | 3%   | CPU     | Low              |
| 6-10 | other       | ~533       | ~3%  | CPU     | Negligible       |

## Root Causes

### 1. PNG write per frame (encode: 13.5ms, 17%)

`annotator.py:78` calls `cv2.imwrite()` for every frame as a PNG, then `release()` reads them all back through ffmpeg. PNG compression is CPU-intensive. On Colab's network-attached disk this adds ~13ms per frame + the ffmpeg re-encode at the end.

**Fix:** Write directly to OpenCV `VideoWriter` with h264 codec. Eliminates disk I/O and double-encode.

### 2. PaddleOCR model load spike (vehicle: 3889ms max)

First plate detection triggers PaddleOCR lazy init (DBNet + CRNN model loading). After that, per-frame cost drops to ~0ms (progressive enrichment caches the plate).

Already fixed — this is a one-time cost.

### 3. YOLO warmup + overhead (detect: 2721ms max, 28ms avg)

First frame includes TensorRT engine warmup. Steady-state is ~15ms but includes tensor copy + NMS. With batch inference across 4 cameras, this could drop to ~25ms total (vs 4 × 28ms = 112ms sequential).

## Multi-Camera Profile (4 feeds, sequential pipeline)

| Metric              | Value              |
|---------------------|--------------------|
| Total time          | 373s               |
| Per-camera FPS      | 1.3 fps            |
| GPU util (avg)      | 25.1%              |
| GPU util (peak)     | 95.0%              |
| CPU util (avg)      | 71.8%              |
| GPU memory          | 1131 / 15360 MiB   |

The GPU is idle 75% of the time. The sequential pipeline processes cameras one-at-a-time, so the GPU alternates between burst (YOLO) and idle (waiting for CPU post-processing). Thread pool helps CPU parallelism but the PNG I/O bottleneck dominates.

## Summary

| Observation | Data |
|-------------|------|
| GPU is the bottleneck? | **No** — 18% util, 82% idle |
| CPU is the bottleneck? | **Yes** — 76% util, 100% peaks |
| I/O is the bottleneck?  | **Yes** — PNG write adds 17% |
| ReID is the bottleneck? | **No** — 13 calls across 500 frames (cached) |
| Memory limited?         | **No** — 366 / 15360 MiB used |

**The single highest-impact fix: switch Annotator from PNG-per-frame + ffmpeg to direct OpenCV h264 VideoWriter. Expected: ~45% reduction in per-frame time.**
