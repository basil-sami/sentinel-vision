# Performance Analysis

## Controlled Benchmark Results (T4, TensorRT YOLO11x)

All tests: 200 frames of "The CCTV People Demo 2.mp4" (640×360, 25fps).
Scripts: `scripts/benchmark.py`, `scripts/bench_detect_split.py`, `scripts/bench_concurrent.py`

---

### Experiment A — detect() Stage Split (CUDA Events)

**Question:** How much of detect() is GPU compute vs CPU post-processing?

| Stage | Avg (ms) | % of detect() |
|-------|----------|---------------|
| Total detect() | **18.65** | 100% |
| GPU compute (CUDA events) | **18.64** | **99.9%** |
| CPU (wall − gpu) | 0.01 | 0.1% |
| Box parsing (Python) | 0.58 | 3.1% |
| Preproc + NMS (overlaps GPU) | −0.56 | — |

**GPU share: 99.9% of detect().** 18.6ms of pure GPU kernel execution per frame.
CPU post-processing is negligible — Ultralytics runs NMS on the GPU inside TensorRT.
The negative "preproc+NMS" means those ops overlap with GPU execution.

**Conclusion: The detector is GPU-bound within detect().** The 18.6ms is pure compute.

---

### Experiment B — Concurrent YOLO Processes (4 workers, 1 GPU)

**Question:** Does the GPU have headroom beyond 53 FPS?

| Config | Total FPS | Per-process FPS | Avg latency | Scaling vs single |
|--------|-----------|-----------------|-------------|-------------------|
| 1 process (baseline) | **53.7** | 53.7 | 18.6 ms | 1.0x (ideal) |
| 4 processes | **51.8** (sum) | ~13 | 77.3 ms | **0.96x** |

**Conclusion: The T4 delivers maximum throughput of ~53 YOLO11x FP16 frames/second
at batch=1.** Four concurrent processes achieve the same aggregate throughput as one,
each getting 1/4 of the GPU time-slice with 4× latency. The GPU has no hidden headroom
for single-frame inference.

---

### Test 1 — GPU Stress (YOLO-only tight loop)

| Metric | Value |
|--------|-------|
| FPS | **53.7** |
| Avg detections/frame | 7.5 |

The earlier 35% GPU utilization from nvidia-smi was an artifact of the profiler's
`subprocess.run(['nvidia-smi'])` sampling consuming CPU time between detect calls.
With clean CUDA event timing the GPU is at full capacity through each detect() call.

---

### Test 2 — No Vehicle Intelligence

| Metric | No Vehicle | Full Pipeline (old ref) | Delta |
|--------|------------|------------------------|-------|
| FPS | **15.2** | 12.7* | +2.5 |
| Speedup | 1.0x | 1.2x* | |

*Old reference included the PNG encode bottleneck (now fixed).

**Conclusion:** Vehicle intelligence adds ~2.5 FPS overhead. Not the dominant problem.

---

### Test 3 — CPU/GPU Overlap Simulation

| Metric | Value |
|--------|-------|
| FPS | **52.7** |
| GPU util (avg) | 80.0% |

GPU utilization reaches 80% with concurrent CPU work. Confirms the GPU can sustain
high utilization — the pipeline just doesn't keep it fed.

---

### Test 4 — Batch Size

| Batch | FPS | Result |
|-------|-----|--------|
| 1 | **54** | Works |
| 2 | — | SKIPPED — TRT engine batch=1 locked |
| 4 | — | SKIPPED |

---

### Test 6 — GPU Idle Gaps

| Metric | Value |
|--------|-------|
| GPU busy ratio (Python thread) | 100% |
| Avg GPU burst | 18.3 ms |
| Avg idle gap between detect calls | 0.0 ms |

Detect calls are back-to-back with zero idle gaps.

---

### Test 7 — CPU/GPU Transfer Cost

| Operation | Time | % of inference |
|-----------|------|----------------|
| Host → Device | 0.29 ms | 1.6% |
| Device → Host | 0.29 ms | 1.6% |
| **Total** | **0.58 ms** | **3.3%** |

Memory transfer is not a bottleneck.

---

## Root Cause Analysis

### The detector is the throughput bottleneck for batch=1

The pipeline budget at batch=1:

| Component | Time | Notes |
|-----------|------|-------|
| detect() | 18.6 ms | GPU, fixed per frame |
| track + vehicle + encode + render | ~16 ms | CPU, estimated after Annator fix |
| **Total per frame** | **~35 ms** | → ~28 FPS ceiling |
| **Measured pipeline** | | → ~15 FPS (CPU scheduling overhead) |

The detector is the largest contributor, but not the only one. Even if YOLO became
infinitely fast, the CPU stages (tracking, vehicle, encoding) would still cap
throughput at ~60 FPS.

### The batching question is unanswered

Batching 4 frames into one TRT call could:

- Keep the GPU busy 4× longer per call (amortizing kernel launch overhead)
- Increase system throughput for multi-camera
- Scale sub-linearly (diminishing returns at larger batches)

**Until a dynamic-batch engine is benchmarked, every throughput estimate is speculative.**

---

## Summary of Findings

| # | Hypothesis | Verdict | Evidence |
|---|-----------|---------|----------|
| 1 | CPU preprocessing is bottleneck | **No** | detect() is 99.9% GPU (CUDA events) |
| 2 | Vehicle analysis is bottleneck | **No** | Removing it: +2.5 FPS (1.2x) |
| 3 | Python postprocessing causes GPU idle | **No** | detect() CPU share: 0.1% |
| 4 | Memory transfer is bottleneck | **No** | 0.58ms total (3.3%) |
| 5 | GPU compute is saturated per frame | **Yes** | 18.6ms pure GPU, 53 FPS ceiling |
| 6 | Batch inference will improve throughput | **Unknown** | Dynamic-batch engine needed |
| 7 | Model downsizing would help | **Unknown** | Benchmark needed |

---

## Next Steps

### Step 1: Build dynamic-batch TensorRT engine

Export YOLO11x with min_batch=1, opt=4, max=8. Then benchmark:

| Batch | FPS | Latency | GPU Memory | GPU Util |
|-------|-----|---------|------------|----------|
| 1 | | | | |
| 2 | | | | |
| 4 | | | | |
| 8 | | | | |

If batch=4 gives >80 FPS → multi-camera batching is the solution.
If batch=4 gives only ~60 FPS → model downsizing should be evaluated.

### Step 2: Benchmark model sizes (if needed)

| Model | GFLOPs | FPS | mAP50 | GPU Memory |
|-------|--------|-----|-------|------------|
| YOLO11m | ~72 | | 72.3 | |
| YOLO11l | ~107 | | 74.0 | |
| YOLO11x | ~195 | 53.7 | 75.3 | |
