# Performance Analysis

## Controlled Benchmark Results (T4, TensorRT YOLO11x)

200 frames of "The CCTV People Demo 2.mp4" (640×360, 25fps).
Script: `scripts/benchmark.py`

---

### Test 1 — GPU Stress (YOLO-only tight loop)

**Question:** Can YOLO keep the T4 busy if we feed it continuously?

| Metric | Value |
|--------|-------|
| FPS | **53.7** |
| GPU util (avg) | **35.0%** |
| GPU util (peak) | 88.0% |
| Avg detections/frame | 7.5 |

**Observation:** Even with zero pipeline overhead, GPU utilization averages 35%.
Each `detect()` call takes ~18.3ms wall time. The GPU compute portion within that
call is unknown without CUDA event timing — it may be the full 18.3ms of GPU work
(just low SM occupancy) or a fraction of it with the rest being CPU postprocessing.

**This is not yet diagnosed.** Test 6 below provides the first clue.

---

### Test 2 — No Vehicle Intelligence

**Question:** Is the vehicle analysis module the bottleneck?

| Metric | Full Pipeline (old ref) | No Vehicle | Delta |
|--------|------------------------|------------|-------|
| FPS | 12.7* | **15.2** | +2.5 |
| Speedup | 1.0x | **1.2x** | |

*Old reference included PNG encode bottleneck, now fixed.

**Conclusion:** Vehicle intelligence is not the primary bottleneck.

---

### Test 3 — CPU/GPU Overlap Simulation

**Question:** Can the GPU reach high utilization with concurrent CPU work?

| Metric | Value |
|--------|-------|
| FPS | **52.7** |
| GPU util (avg) | **80.0%** |
| GPU busy ratio (Python thread) | 100% |

**Conclusion:** GPU utilization jumps from **35% → 80%** when CPU work runs concurrently
on a separate thread. The GPU is capable of sustained utilization. The current
serial pipeline is preventing this.

---

### Test 4 — Batch Size

| Batch | FPS | Result |
|-------|-----|--------|
| 1 | **54** | Works |
| 2 | — | **SKIPPED** — TRT engine batch=1 locked |
| 4 | — | **SKIPPED** — TRT engine batch=1 locked |

**Action needed:** Rebuild TensorRT engine with dynamic batch (min=1, opt=4, max=8).

---

### Test 6 — GPU Idle Gaps

| Metric | Value |
|--------|-------|
| GPU busy time (Python perspective) | 1832 ms |
| Total wall time | 1832 ms |
| GPU busy ratio | **100%** (detect calls are back-to-back) |
| Avg GPU burst | **18.3 ms** |
| Avg idle gap between detect calls | **0.0 ms** |

**Important:** The Python thread submits `detect()` calls back-to-back with zero idle
gaps. The nvidia-smi reading of 35% GPU utilization means the GPU is active 35% of
the wall-clock time **within** each `detect()` call. The remaining 65% of each
`detect()` call is CPU work (preprocessing, NMS, box decoding, Python overhead).

This is the key insight: **GPU is executing kernels efficiently but is idle while
Python processes results between frames.**

---

### Test 7 — CPU/GPU Transfer Cost

| Operation | Time | % of inference |
|-----------|------|----------------|
| Host → Device (640×480 RGB) | **0.29 ms** | 1.6% |
| Device → Host | **0.29 ms** | 1.6% |
| Full inference round-trip | **17.9 ms** | |
| Transfer total | **0.58 ms** | **3.3%** |

**Conclusion:** Memory transfer is not a bottleneck. Pinned memory, zero-copy,
and CUDA streams optimizations are unnecessary at this stage.

---

### Summary of Findings

| # | Hypothesis | Verdict | Evidence |
|---|-----------|---------|----------|
| 1 | CPU preprocessing is the bottleneck | **Not primary** | GPU-only loop achieves same 35% util |
| 2 | Vehicle analysis is the bottleneck | **No** | Removing it: +2.5 FPS (1.2x) |
| 3 | Python GIL + sync postprocessing serializes GPU | **Yes** | Overlap test: 35% → 80% GPU util |
| 4 | Memory transfer is a bottleneck | **No** | 0.58ms total (3.3% of inference) |
| 5 | Low GPU kernel occupancy | **Unknown** | Need CUDA event timing inside detect() |
| 6 | Insufficient batching | **Cannot test** | TRT engine batch=1 locked |

---

### Root Cause Analysis

**The pipeline serializes CPU postprocessing and GPU inference, creating idle bubbles.**

Current timeline (single frame):

```
┌─────────────────────────────────────────┐
│ detect() — 18.3 ms wall time             │
│                                           │
│  ┌──────────┐  ┌──────────────────────┐  │
│  │ GPU exec  │  │ CPU: NMS, box       │  │
│  │           │  │ decode, Python       │  │
│  │ ??? ms    │  │ ??? ms               │  │
│  └──────────┘  └──────────────────────┘  │
└─────────────────────────────────────────┘
                                          ┌──────────────────────────┐
                                          │ Next detect() — 18.3 ms  │
                                          │ GPU waits for CPU finish  │
                                          └──────────────────────────┘
```

**The exact split between GPU compute and CPU postprocessing inside `detect()`
is unknown.** This is the next measurement to take.

---

### Next Experiments

| # | Experiment | Method | Expected Outcome |
|---|-----------|--------|-----------------|
| A | **Split detect() timing** | CUDA events before/after predict() to isolate GPU compute from CPU postprocess | Identify whether GPU time is 6ms or 16ms per call |
| B | **Concurrent YOLO processes** | 4 processes sharing 1 GPU via multiprocessing | Measures real batch-equivalent throughput; GPU util reveals headroom |
| C | **Rebuild TRT dynamic batch** | Export with min=1, opt=4, max=8 | Enables `detect_batch()` for multi-camera |


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

**Total: 78.9 ms/frame → 12.7 FPS (old PNG-based Annotator)**

### System Utilization

| Metric         | Value              | Verdict                    |
|----------------|--------------------|----------------------------|
| GPU util (avg) | 17.9%              | **GPU idle 82% of time**   |
| GPU util (peak)| 54.0%              | Never fully loaded          |
| GPU memory     | 366 / 15360 MiB   | Barely using it             |
| CPU util (avg) | 75.9%              | CPU saturated               |
| CPU util (peak)| 100.0%             | Hitting the ceiling         |
| ReID calls     | 13 (across 500 fr) | Caching works effectively   |

### Bottleneck Ranking (old pipeline)

| Rank | Component   | Total (ms) | %    | CPU/GPU | Expected Speedup |
|------|-------------|------------|------|---------|-----------------|
| 1    | detect      | 14015      | 36%  | GPU     | Medium (batch)   |
| 2    | vehicle     | 13793      | 35%  | GPU/CPU | Low (one-time)   |
| 3    | **encode**  | **6758**   | **17%** | **I/O** | **Fixed**        |
| 4    | track       | 2876       | 7%   | CPU     | Low              |
| 5    | render      | 956        | 3%   | CPU     | Low              |
| 6-10 | other       | ~533       | ~3%  | CPU     | Negligible       |

## Multi-Camera Profile (4 feeds, sequential pipeline)

| Metric              | Value              |
|---------------------|--------------------|
| Total time          | 373s               |
| Per-camera FPS      | 1.3 fps            |
| GPU util (avg)      | 25.1%              |
| GPU util (peak)     | 95.0%              |
| CPU util (avg)      | 71.8%              |
| GPU memory          | 1131 / 15360 MiB   |

## Revised Roadmap

| Priority | Task | Confidence |
|----------|------|-----------|
| 1 | Pipeline CPU/GPU overlap (producer-consumer) | Very High |
| 2 | Rebuild TensorRT engine with dynamic batch (min=1, opt=4, max=8) | Very High |
| 3 | Shared detector for all cameras | Very High |
| 4 | Split detector timing: GPU execution vs CPU postprocessing | High |
| 5 | Vehicle micro-profiling | Medium |
| 6 | CPU worker scaling if hardware allows | Medium |
