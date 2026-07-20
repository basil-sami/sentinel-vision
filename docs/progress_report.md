# Sentinel Vision — Progress Report

**Branch:** `phase-4-vehicle-intelligence`

---

## Phase 1 — Detection & Tracking (complete)

- YOLO11 / YOLO12 / RT-DETR detector (`model_family` param for zero-code swap)
- BoT-SORT tracker with ReID (`osnet_x1_0_msmt17`, upgraded from `x0_25`)
- 62 COCO classes (expanded from 6: backpack, suitcase, handbag, laptop, cell phone, book, knife, etc.)
- Configurable thresholds: `conf_threshold`, `track_thresh`, `match_thresh`, `track_buffer`
- TensorRT auto-export (`.pt` → `.engine`, 2–5× speedup)
- Numba JIT for zone containment + gate crossing (10–50×, optional dep)
- Multi-stream: `CameraWorker(Process)` per camera, `MultiCameraPipeline`

## Phase 2 — Zones, Gates & Core Analytics (complete)

- Zone entry/exit events with polygon containment (ray-casting)
- Gate crossing with directional detection (entering/exiting)
- Loitering detection (configurable dwell threshold, confidence = dwell / 1200s)
- Abandoned object detection (non-person tracks stationary beyond buffer)
- Dwell tracking per zone, gate counting (entries/exits/net)
- Calibration (pixel ↔ meter homography, 4+ point correspondences)
- Interaction model (group/traveling behavior)
- Evidence capture (clip recording per high-severity event)
- Stationary false-positive filter (<20px movement, ≥5 frames)

## Phase 3 — Architecture & Infrastructure (complete)

- **Track State Machine:** NEW → ACTIVE ↔ OCCLUDED → LOST → MERGED → ENDED with validated transitions
- **Identity Confidence:** per-track avg/min confidence, stability, appearance variance, occlusion count
- **Analytics DB:** SQLite with 6 tables (cameras, runs, tracks, events, incidents, gate_counts), batch inserts/queries
- **Event Correlation:** 4 incident rules — suspicious_activity, gate_breach, object_drop, prolonged_loitering
- **Track Predictor:** Kalman constant-velocity filter, 4-state, confidence decay
- **Plugin Architecture:** `AnalyticsPlugin` ABC (`initialize`/`process_frame`/`process_track`/`process_event`/`shutdown`) + `PluginRegistry`
- **Benchmark Suite:** `SpeedTracker` (instant/overall FPS), `compute_mot_metrics` (TP/FP/FN/ID switches/IoU)
- **Event Bus:** pub/sub with wildcard topics, priority levels, history buffer
- **Camera Abstraction:** `Camera` dataclass (id/name/source/gps/location/topology/status) + `CameraRegistry` (JSON save/load)
- **Time Sync:** `FrameTimestamp` with UTC ISO time per frame
- **Config System:** YAML defaults (`configs/defaults/{detector,tracker,analytics,camera,vehicle}.yaml`), `ConfigLoader` with deep-merge
- **Unit Tests:** 23 tests covering all modules with synthetic video generation

## Phase 4 — Vehicle Intelligence (complete)

- **Plate Detector:** PaddleOCR DBNet primary, contour fallback
- **Plate Reader:** PP-OCR recognition with 3 preprocessing variants (raw, sharpen, CLAHE contrast)
- **Validation:** regex per region (US/UK/EU/generic), charset filter, length bounds
- **Temporal Fusion:** 10+ frame buffer, confidence-weighted scoring
- **Vehicle Registry:** persistent identity keyed by plate or color+type, tracks visits/parking/history
- **Vehicle Events:** speeding, vehicle_parking, plate_read, repeat_visitor
- **Vehicle Attributes:** color extraction, size classification (small/medium/large)
- **Orchestrator:** `VehicleAnalyzer` integrated into pipeline frame loop with calibrator speed checks
- End-to-end confirmed: 200 frames, 12 tracks, 694 detections, 17 events, 10 vehicles, gate counts generated

## Pre-Phase 5 — Scene Understanding (complete)

- **Carrying Detection** (`src/analytics/scene/carrying.py`)
  - Person-object spatial relationship (IoU > 5% or centroid containment)
  - Co-motion verification (velocity cosine similarity > 0.5 across 5+ frames)
  - Carryable classes: backpack, handbag, suitcase, laptop, cell phone, book, bottle, cup, umbrella, knife
  - Event: `person_carrying` (medium severity)

- **Overloaded Vehicle Detection** (`src/analytics/scene/overloaded_vehicle.py`)
  - Persons inside vehicle bbox (15% margin) moving at same velocity
  - Threshold: 3+ persons per vehicle (truck/bus/car/motorcycle)
  - One-shot emission per vehicle (no repeat alerts)
  - Event: `overloaded_vehicle` (high severity)

- **Scene Analyzer** (`src/analytics/scene/orchestrator.py`)
  - Standard `process_frame(tracks, frame_index, calibrator, zone_mgr)` hook
  - Wired into pipeline after vehicle intelligence block

---

## Quick Reference

| What | Where |
|------|-------|
| Pipeline | `src/pipeline.py` — `analyze_video()` |
| Detector | `src/detection/yolo_detector.py` — `YOLODetector` |
| Tracker | `src/tracking/tracker.py` — `Tracker` |
| Zones/Gates | `src/analytics/zones.py` — `ZoneManager` |
| Calibration | `src/analytics/calibration.py` — `Calibrator` |
| Vehicle ANPR | `src/analytics/vehicle/orchestrator.py` — `VehicleAnalyzer` |
| Scene Intelligence | `src/analytics/scene/orchestrator.py` — `SceneAnalyzer` |
| Event Model | `src/models/event.py` — `Event`, `EventStore` |
| Event Bus | `src/events/bus.py` — `EventBus` |
| Config | `src/config.py` — `ConfigLoader` |
| Camera Model | `src/models/camera.py` — `Camera`, `CameraRegistry` |
| Analytics DB | `src/db/repository.py` — `AnalyticsDB` |
| TensorRT | `src/optimization/tensorrt_export.py` |
| Multi-Stream | `src/optimization/multi_stream.py` — `MultiCameraPipeline` |
| Tests | `tests/test_synthetic.py` — 23 tests |
| Notebook | `notebooks/sentinel_demo.ipynb` — Colab demo |

---

## Known Gaps

- YOLO (COCO) detects only `knife` (class 43), not guns/rifles — needs fine-tuning on weapon dataset
- Uniform/personnel classification not yet implemented (hook exists for zone-based authorization)
- Multi-camera person re-ID across cameras not yet implemented
- Automatic zone learning from scene context not yet implemented
