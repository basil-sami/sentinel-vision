# Performance Analysis

## Observed Behavior

Single camera, Colab T4, YOLO12x TensorRT:

| Phase | Frames | Speed | Note |
|-------|--------|-------|------|
| Empty shot (0-100) | 0 detections | 46 it/s | No ReID, no ANPR |
| Active (100+) | 5-8 persons | 12 it/s | ReID on every person every frame |
| Cache miss (first vehicle) | ~1 frame | ~2-3 s stall | PaddleOCR lazy-loads recognition model |

## Per-Frame Cost Breakdown (5-8 persons, 1 vehicle)

| Step | Device | Cost | % of 83ms |
|------|--------|------|------------|
| YOLO12x TensorRT | GPU | 8 ms | 10% |
| ReID OSNet x1_0 (×5-8 persons) | GPU | 75-200 ms | 90-96% |
| PaddleOCR text rec (×1 vehicle, every 10th frame) | GPU | 10-30 ms | intermittent |
| Tracker + matching | CPU | 3 ms | 4% |
| Color extraction (k-means, every frame) | CPU | 2-5 ms | 2-6% |
| Scene + interaction O(N²) | CPU | 1-2 ms | 1-2% |
| Rendering + write | CPU | 2-3 ms | 2-4% |

The GPU does **two separate forward passes per detection**:
1. YOLO → once per frame for the whole image
2. ReID → once per detection (sequential, not batched)

## Bottleneck Summary

- **Primary: ReID** — 90%+ of per-frame time. Each person crop is a separate GPU forward pass.
- **Secondary: ANPR** — Lazy model load stalls first vehicle frame. Subsequent reads are fast.
- **Tertiary: Per-frame per-vehicle color k-means** — unnecessary recompute (vehicle color is constant).

## Scaling Projections

| Cameras | Approach | Per-camera FPS | Total Throughput |
|---------|----------|---------------|------------------|
| 1 | Sequential | ~12 fps | 12 fps |
| 4 | Sequential | ~3 fps | 12 fps |
| 4 | Thread pool (CPU parallel) | ~12 fps | 48 fps |
| 20 | Thread pool + batch ReID | ~10-12 fps | 200+ fps |

## Fix Candidates (ordered by impact)

1. **Batch ReID** — Stack all person crops across all cameras → single forward pass (batch=10-40). ~3× faster than sequential ReID per-frame.

2. **Thread pool** — YOLO inference on main thread → dispatch per-camera post-processing to CPU workers (tracker match, zones, scene, rendering). Embeddings computed once, shared.

3. **Cache vehicle attributes** — Compute `extract_vehicle_color()` once on first appearance, reuse thereafter.

4. **Reduce ANPR frequency** — Current `plate_read_interval=10` (read every 10th frame). Already tuned.
