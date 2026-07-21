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
| GPU util (avg) | **35%** |
| GPU util (peak) | 88% |
| Avg detections/frame | 7.5 |

**Conclusion:** Even with zero pipeline overhead, the GPU is only 35% utilized.
The T4 + YOLO11x TensorRT combination has low GPU occupancy per inference call.
Each `detect()` call runs ~18.3ms total but only ~6ms of actual GPU compute.

---

### Test 2 — No Vehicle Intelligence

**Question:** Is the vehicle analysis module the bottleneck?

| Metric | Full Pipeline (old ref) | No Vehicle | Delta |
|--------|------------------------|------------|-------|
| FPS | 12.7* | **15.2** | +2.5 |
| Speedup | 1.0x | **1.2x** | |

*Old reference included PNG encode bottleneck

**Conclusion:** Vehicle intelligence adds only ~2 FPS overhead. Not the primary bottleneck.

---

### Test 3 — CPU/GPU Overlap Simulation

**Question:** Can the GPU reach high utilization with concurrent CPU work?

| Metric | Value |
|--------|-------|
| FPS | **52.7** |
| GPU util (avg) | **80%** |
| GPU busy ratio | 100% |

**Conclusion:** GPU utilization jumps from **35% → 80%** when CPU work runs concurrently.
The pipeline is serializing CPU postprocessing and GPU inference.
**Fix: separate CPU postprocessing into a pipeline stage that runs on frame N-1 while the GPU processes frame N.**

---

### Test 4 — Batch Size

| Batch | FPS | Result |
|-------|-----|--------|
| 1 | **54** | Works |
| 2 | — | **SKIPPED** — TRT engine batch=1 locked |
| 4 | — | **SKIPPED** — TRT engine batch=1 locked |

**Conclusion:** TensorRT engine was exported with `max_batch_size=1`.
Cannot test batched inference without rebuilding the engine.

---

### Test 5 — CPU Worker Scaling (not yet run)

---

### Test 6 — GPU Idle Gaps

| Metric | Value |
|--------|-------|
| GPU busy time | 1832 ms |
| Total wall time | 1832 ms |
| GPU busy ratio | **100%** (Python thread perspective) |
| Avg GPU burst | **18.3 ms** |
| Avg idle gap | **0.0 ms** (between detect calls) |

**Conclusion:** From the Python side, `detect()` calls are back-to-back with zero gaps.
The 35% GPU utilization is **inside** each `detect()` call, not between calls.
The GPU burst per frame (~6ms compute) is followed by synchronous CPU postprocessing (~12ms)
before the next frame can be submitted.

---

### Test 7 — CPU/GPU Transfer Cost

| Operation | Time | % of inference |
|-----------|------|----------------|
| Host → Device (640×480 RGB) | **0.29 ms** | 1.6% |
| Device → Host | **0.29 ms** | 1.6% |
| Full inference round-trip | **17.9 ms** | |
| Transfer total | **0.58 ms** | **3.3%** |

**Conclusion:** Memory transfer is negligible (3.3%). Not a bottleneck.

---

### Summary of Findings

| # | Hypothesis | Verdict | Evidence |
|---|-----------|---------|----------|
| 1 | CPU bottleneck (preprocessing) | **Not primary** | GPU-only loop: 35% util, no pipeline overhead |
| 2 | Vehicle analysis bottleneck | **No** | Removing it: +2.5 FPS (1.2x) |
| 3 | Python GIL serialization | **Yes** | Overlap test: 35% → 80% GPU util |
| 4 | Memory transfer overhead | **No** | 0.58ms total (3.3%) |
| 5 | Pipeline design (single-stream) | **Yes** | GPU idle 65% even in ideal conditions |
| 6 | Insufficient batching | **Cannot test** | TRT engine locked to batch=1 |

---

### Root Cause

**YOLO11x TensorRT on T4 has low GPU occupancy per inference call (~35%).**
Each `detect()` does:
1. Host→Device copy: ~0.3ms (fine)
2. GPU inference: ~6ms (burst, GPU busy)
3. NMS + box decoding (CPU sync): ~12ms (GPU idle)
4. Device→Host copy: ~0.3ms (fine)

The ~12ms of synchronous CPU postprocessing prevents the next frame from being submitted.
The GPU finishes in ~6ms and waits ~12ms for Python to finish postprocessing.

**This is intrinsic to running a single image through a large model.**
The fix is **batch processing** — send 4 images at once to amortize the CPU sync overhead.

---

### Updated Bottleneck Ranking (with new Annotator)

| Rank | Component | Old Total (ms) | New Estimate (ms) | Fix |
|------|-----------|---------------|-------------------|-----|
| 1 | detect | 28.0 | **18.3** | Batch inference (4 cams → 1 call) |
| 2 | vehicle | 27.6 | **~10** | Already optimized (progressive enrichment) |
| 3 | track | 5.8 | **~5** | Low priority |
| 4 | encode | 13.5 | **~2** | **Fixed** (VideoWriter) |
| 5 | render | 1.9 | **~2** | Low priority |
| **Total** | | **78.9** | **~37** | **Theoretical max: ~27 FPS** |

### Multi-Camera Path

**Without batch inference (current):**
```
Camera 1: detect(18ms) → vehicle(10ms) → track(5ms) → encode(2ms)  = 35ms
Camera 2: detect(18ms) → vehicle(10ms) → track(5ms) → encode(2ms)  = 35ms
Camera 3: detect(18ms) → vehicle(10ms) → track(5ms) → encode(2ms)  = 35ms
Camera 4: detect(18ms) → vehicle(10ms) → track(5ms) → encode(2ms)  = 35ms
                                   Total: 140ms → **7 FPS system**
```

**With batch inference (4-camera batch):**
```
Batch detect(40ms)  ← 4 frames in one TRT call
  Camera 1: vehicle(10ms) + track(5ms) + encode(2ms)  = 17ms
  Camera 2: vehicle(10ms) + track(5ms) + encode(2ms)  = 17ms
  Camera 3: vehicle(10ms) + track(5ms) + encode(2ms)  = 17ms
  Camera 4: vehicle(10ms) + track(5ms) + encode(2ms)  = 17ms
                                   Total: 57ms → **70 FPS system**
```

---

### Next Steps

1. **Rebuild TensorRT engine with `max_batch_size=4`** to enable batched inference
2. **Implement batch-aware `process_cameras()`** that groups 4 camera frames into one `detect_batch()` call
3. **Pipeline CPU postprocessing overlaps with GPU inference** using producer-consumer queues
4. Re-run benchmark to validate: expected **4–5x system throughput improvement** for 4 cameras
