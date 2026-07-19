# Sentinel Vision — Phase 2 Report

## Objective
Reduce ID switching / track fragmentation in multi-object tracking for a fixed CCTV surveillance video.

## Baseline (ByteTrack only)
- **Tracker:** ByteTrack (`track_thresh=0.5`, `match_thresh=0.9`, `track_buffer=30`)
- **Detector:** YOLO11nano
- **Video:** `The CCTV People Demo 2.mp4` — 640x360, 1932 frames, 25fps, ~77s
- **Result:** 91 unique person IDs (estimated <50 actual people in scene)
- **Cause:** Motion-only (IoU) matching loses identity on occlusion and crossing.

## Solution — BoT-SORT + ReID

Replaced ByteTrack with BoT-SORT, which adds appearance-based re-identification via a learned embedding model.

### Tracker config (final)
| Parameter | Value | Purpose |
|---|---|---|
| Backend | BoT-SORT (boxmot v19) | IoU + appearance fusion |
| ReID model | osnet_x0_25_msmt17.pt (3MB) | 256-D appearance embedding |
| track_high_thresh | 0.5 | Initiate new tracks |
| track_low_thresh | 0.1 | Continue existing tracks (partial occlusion) |
| track_buffer | 300 frames (12s) | Survive long occlusions |
| match_thresh | 0.8 | Association gate |
| cmc_method | ecc (default) | Camera-motion comp |
| fuse_first_associate | False | Motion only in first pass |

### Detector config (final)
| Parameter | Value | Purpose |
|---|---|---|
| Model | YOLO11x (xlarge, 109MB) | Best detection quality |
| conf_threshold | 0.25 | Catch partially occluded people |
| Device | Tesla T4 (Colab) | ~7-8 min for full video |

### Post-processing
- **merge_fragments.py** — merges track fragments of the same class where gap < track_buffer frames, centroid distance < 100px at the junction, and no temporal overlap.

### Results
- **83 unique tracked objects** across all classes (63 person, 15 car, 3 bus, 2 truck)
- Reduction from 91 to 63 person IDs (~31% fewer fragments)
- Remaining fragmentation due to very long occlusions (>buffer) and appearance changes

## Files Changed
- `src/tracking/tracker.py` — ByteTrack to BoT-SORT; add `use_reid`, `track_low_thresh`, `torch.device` fix
- `src/pipeline.py` — Pass `use_reid`, `device` to tracker; add `merge_fragments` call
- `src/detection/yolo_detector.py` — Add `large` / `xlarge` model sizes
- `src/analytics/merge_fragments.py` — New post-merge spatial/temporal heuristic
- `notebooks/sentinel_demo.ipynb` — Use ReID, xlarge model, 300 buffer
- `.gitignore` — Add model weight files

## Next Steps (Phase 3)
- Zone-based counting (entry/exit)
- Loitering / dwell-time detection
- Abandoned object detection
- Movement analytics (direction, speed, heatmaps)
- Event-based alerting
