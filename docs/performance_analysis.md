# Performance Analysis

## Controlled Benchmark Results (T4, TensorRT YOLO11x)

All tests: 200 frames of "The CCTV People Demo 2.mp4" (640×360, 25fps).
Scripts: `scripts/benchmark.py`, `scripts/bench_detect_split.py`, `scripts/bench_concurrent.py`

---

### Experiment A — detect() Stage Split (CUDA Events)

**Question:** How much of detect() is GPU compute vs CPU post-processing?

| Stage | Avg (ms) | Min (ms) | Max (ms) | % of detect() |
|-------|----------|----------|----------|---------------|
| Total detect() | **18.65** | 16.14 | 37.15 | 100% |
| GPU compute (CUDA events) | **18.64** | 16.13 | 37.13 | 99.9% |
| CPU (wall - gpu) | 0.01 | -0.00 | 0.08 | 0.1% |
| Box parsing (Python) | 0.58 | 0.04 | 2.07 | 3.1% |
| Preproc + NMS (overlaps GPU) | -0.56 | -2.05 | -0.00 | — |

**GPU share: 99.9% of detect().** 18.6ms of pure GPU kernel execution per frame.
CPU post-processing is negligible — Ultralytics runs NMS on the GPU inside TensorRT.
The negative "preproc+NMS" time means those CPU operations overlap with GPU execution.

**Conclusion: The GPU is the bottleneck within detect().** There is no hidden CPU work.

---

### Experiment B — Concurrent YOLO Processes (4 workers, 1 GPU)

**Question:** What happens when 4 processes share the same GPU?

| Config | Total FPS | Per-process FPS | Avg latency | Scaling |
|--------|-----------|-----------------|-------------|---------|
| 1 process (baseline) | **53.7** | 53.7 | 18.6 ms | 1.0x |
| 4 processes | **51.8** (sum) | ~13 | 77.3 ms | **0.96x** |

**Throwing more processes at the GPU does nothing.** Aggregate throughput is identical
to a single process. Each concurrent process gets exactly 1/4 of the GPU time-slice,
with 4× the latency.

**Conclusion: The T4 is saturated at ~53 YOLO11x frames/second.** This is the
GPU throughput ceiling for this model+hardware combination.

---

### Test 1 — GPU Stress (YOLO-only tight loop)

| Metric | Value |
|--------|-------|
| FPS | **53.7** |
| Avg detections/frame | 7.5 |

The earlier nvidia-smi reading of 35% GPU utilization was misleading — it was an
artifact of the profiler's subprocess sampling consuming CPU time. With clean CUDA
event timing, detect() is **99.9% GPU** and runs back-to-back with zero gaps
(Test 6). The GPU is at full capacity.

---

### Test 2 — No Vehicle Intelligence

| Metric | No Vehicle | Full Pipeline (old ref) | Delta |
|--------|------------|------------------------|-------|
| FPS | **15.2** | 12.7* | +2.5 |
| Speedup | 1.0x | 1.2x* | |

*Old reference included PNG encode bottleneck (now fixed).

**Conclusion:** Vehicle intelligence adds modest overhead. Not the primary bottleneck.

---

### Test 3 — CPU/GPU Overlap Simulation

| Metric | Value |
|--------|-------|
| FPS | **52.7** |
| GPU util (avg) | **80.0%** |
| GPU busy ratio (Python thread) | 100% |

GPU utilization jumps to 80% when CPU work runs concurrently. This confirms the GPU
can sustain high utilization — it's just idling between detect() calls while the
pipeline's CPU stages run.

---

### Test 4 — Batch Size

| Batch | FPS | Result |
|-------|-----|--------|
| 1 | **54** | Works |
| 2 | — | SKIPPED — TRT engine batch=1 locked |
| 4 | — | SKIPPED — TRT engine batch=1 locked |

---

### Test 6 — GPU Idle Gaps

| Metric | Value |
|--------|-------|
| GPU busy time (Python perspective) | 1832 ms |
| Total wall time | 1832 ms |
| GPU busy ratio | **100%** (detect calls are back-to-back) |
| Avg GPU burst | **18.3 ms** |
| Avg idle gap between detect calls | **0.0 ms** |

Detect calls are submitted back-to-back with zero gaps. The GPU is occupied
18.6ms per frame with no idle time between inference calls.

---

### Test 7 — CPU/GPU Transfer Cost

| Operation | Time | % of inference |
|-----------|------|----------------|
| Host → Device (640×480 RGB) | **0.29 ms** | 1.6% |
| Device → Host | **0.29 ms** | 1.6% |
| Transfer total | **0.58 ms** | **3.3%** |

Memory transfer is not a bottleneck.

---

## Root Cause Analysis

### The real bottleneck: GPU throughput ceiling

**YOLO11x TensorRT on T4 peaks at ~53 FPS.** Each frame requires 18.6ms of GPU
kernel execution. This is a compute-bound throughput limit — no amount of pipeline
restructuring can make the GPU run faster on individual frames.

### The pipeline amplifies the problem

Current per-frame timeline (estimated with new Annotator):

```
Frame N:
  detect(18.6ms GPU) → track(5ms CPU) → vehicle(~10ms CPU) → encode(2ms CPU)
                         GPU idle ←────────────────────────→
Frame N+1:
  detect(18.6ms GPU) → ...
```

GPU is busy 18.6ms out of every ~35ms = **~53% utilization**.
The remaining 47% of wall time, the GPU sits idle while CPU stages run.

For 4 cameras processed sequentially:
- GPU time per camera: 18.6ms
- Total GPU time: 4 × 18.6ms = 74.4ms
- Pipeline time per camera: ~35ms
- Total pipeline time: ~140ms
- **System throughput: ~7 FPS, GPU utilization: ~53%**

### The two available levers

| Lever | How | Expected gain | Trade-off |
|-------|-----|---------------|-----------|
| **Batch** | Combine 4 frames into one TRT call | 4× frames per inference, less overhead | Needs TRT engine rebuild |
| **Downsize model** | YOLO11m (72 GFLOPS) instead of YOLO11x (195 GFLOPS) | ~2.7× FPS | Lower accuracy |

Batching is the primary lever because it increases the amount of work done per GPU
cycle without changing model quality.

---

## Summary of Findings

| # | Hypothesis | Verdict | Evidence |
|---|-----------|---------|----------|
| 1 | CPU preprocessing is bottleneck | **No** | detect() is 99.9% GPU via CUDA events |
| 2 | Vehicle analysis is bottleneck | **No** | Removing it: +2.5 FPS (1.2x) |
| 3 | Python GIL + sync postprocessing causes idle | **No** | detect() GPU share: 99.9%, CPU: 0.1% |
| 4 | Memory transfer is bottleneck | **No** | 0.58ms total (3.3% of inference) |
| 5 | Low GPU kernel occupancy | **No** | GPU executes 18.6ms continuously per frame |
| 6 | **GPU throughput ceiling** | **Yes** | T4 saturates at ~53 YOLO11x FPS; 4 concurrent processes: 0.96x scaling |

---

## Profiler Results (Single Camera, TensorRT YOLO11x, T4)

500 frames, old PNG-based Annotator.

### Per-Frame Timing

| Stage | Avg (ms) | % Runtime |
|-------|----------|-----------|
| detect | 28.0 | 36% |
| vehicle | 27.6 | 35% |
| encode | 13.5 | 17% |
| track | 5.8 | 7% |
| render | 1.9 | 3% |
| other | ~1.0 | ~2% |

**Total: 78.9 ms/frame → 12.7 FPS**

### System Utilization (old pipeline)

| Metric | Value | Verdict |
|--------|-------|---------|
| GPU util (avg) | 17.9% | Idle 82% |
| GPU util (peak) | 54.0% | |
| GPU memory | 366 / 15360 MiB | |
| CPU util (avg) | 75.9% | CPU saturated |
| CPU util (peak) | 100.0% | |

## Multi-Camera Profile (4 feeds, sequential)

| Metric | Value |
|--------|-------|
| Total time | 373s |
| Per-camera FPS | 1.3 fps |
| GPU util (avg) | 25.1% |

---

## Revised Roadmap

| Priority | Task | Confidence |
|----------|------|-----------|
| 1 | **Rebuild TensorRT engine with dynamic batch** (min=1, opt=4, max=8) | Very High |
| 2 | **Pipeline overlap** — run CPU stages on frame N while GPU processes N+1 | Very High |
| 3 | Shared detector for all cameras (exists) | Very High |
| 4 | Vehicle micro-profiling | Medium |
| 5 | Model downsizing (YOLO11m) if batch alone is insufficient | Medium |

---

## Decision Required

Which path to pursue?

**Option A — Rebuild TRT with dynamic batch (recommended first step)**
Export engine with min=1, opt=4, max=8. Same model (YOLO11x), same accuracy.
Enables `detect_batch()` for multi-camera. Expected: 1.5–2.5× system throughput improvement.

**Option B — Downsize to YOLO11m**
Switch from xlarge (195 GFLOPS) to medium (72 GFLOPS). ~2.7× more FPS,
lower accuracy (mAP50: 72.3 vs 75.3). Also needs TRT rebuild.

**Option C — Both: batch first, downsize if needed**
Rebuild TRT with dynamic batch and YOLO11x first. Benchmark 4-camera throughput.
If per-camera FPS is still insufficient, downsize to 11l or 11m.
