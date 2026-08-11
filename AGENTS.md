# Sentinel Vision — Module Index

## Architecture Overview

```
Video/Camera → VideoLoader → YOLODetector → Tracker (BoT-SORT+ReID) → Per-frame Analytics → Post-process → Output
                                                                           │
                                                                      MultiCameraPipeline
                                                                      (shared detector,
                                                                       per-camera tracker)
```

53 source files across 11 packages. No stubs — every module is a real implementation.

---

## Entry Points

### `scripts/bench_realtime.py` — Live camera simulator + latency benchmark
**Purpose**: Simulates N camera feeds, measures pipeline lag vs real-time.
**Key args**: `--multi` (4 cams), `--speed` (multiplier), `--model-size`, `--tensorrt`, `--output-csv`
**Architecture**: `CameraSimulator` class per camera — threads (not thread pool), standalone pipeline (not `analyze_video`), no video output.
**⚠️ Hardcoded `use_reid=True`** (was `False` before Phase 4.1). Still missing: post-merge, stationary filter, zone logic, scene analysis, video encoding. Latency ~30% lower than production.

### `scripts/profile_pipeline.py` — Full pipeline profiler
**Purpose**: Runs `analyze_video` with `PipelineProfiler` wrapping every stage. Closest to production fidelity.
**Key args**: `--model`, `--tensorrt`, `--multi`, `--frames`

### `scripts/benchmark.py` — GPU bottleneck diagnostic (7 experiments)
**Purpose**: GPU stress, vehicle disabling, overlap analysis, batch scaling, worker scaling, idle gap, transfer cost.
**⚠️ Uses `use_reid=False`** internally.

### `scripts/eval_fanvid.py` — FANVID face + LP benchmark
**Purpose**: Tests face recognition and license plate accuracy on low-res (180×320) FANVID clips.
**Two modes**: `direct` (insightface on raw frames, 2× upscaled) vs `pipeline` (full YOLO→track→face).
**Key flags**: `--mode`, `--match-threshold` (0.30 default), `--min-face-size` (20), `--num-videos`

### `scripts/test_4feeds.py` — Hardcoded 4-camera test
**Purpose**: Runs `MultiCameraPipeline` on 4 transformed feeds, generates mosaic. No argparse.

### `scripts/face_db_matcher.py` — Unknown face batch matcher
**Purpose**: Batch-match unknown captures against known gallery, auto-promote high-confidence matches.

---

## Detection — `src/detection/yolo_detector.py`

**File**: `src/detection/yolo_detector.py` (230 lines)
**Class**: `YOLODetector`

| Method | Lines | Purpose |
|--------|-------|---------|
| `__init__()` | 118-154 | Loads YOLO (PyTorch or TensorRT engine). Graceful fallback: TRT→PT |
| `shared()` | 156-170 | Singleton cache keyed by `model_family:model_size:device:use_tensorrt` (⚠️ omits `target_classes`, `tensorrt_half`, `batch_size`) |
| `detect_batch()` | 178-203 | Takes `list[np.ndarray]`, returns `list[list[Detection]]`. Used only by bench scripts, NOT by pipeline |
| `detect()` | 205-228 | Single frame → `list[Detection]`. Called by pipeline every frame. Filtered by `target_classes` (62 surveillance classes) |
| `detect_batch()` | 178-203 | Same logic as detect() but list input. Code duplication with detect() |

**Detection dataclass** (line 14): `class_id`, `class_name`, `confidence`, `bbox: (x1,y1,x2,y2)`, `to_dict()`

**`COCO_TARGET_CLASSES`** (line 84): dict of 62 surveillance-relevant COCO class IDs. Applied as post-inference filter.

**Critical**: `self.batch_size` stored but never read by `detect()` or `detect_batch()`. Pipeline calls `detect()` per frame — never batches.

## TensorRT — `src/optimization/tensorrt_export.py`

**File**: `src/optimization/tensorrt_export.py` (73 lines)

| Function | Purpose |
|----------|---------|
| `engine_path()` | Returns filename like `yolo11x_b4.engine`. ⚠️ `half` param not embedded in filename → FP16/FP32 collision |
| `has_engine()` | Checks engine file exists |
| `export_to_engine()` | Exports `.pt` → `.engine` via `YOLO.export()`. Supports dynamic batch. ⚠️ Rename logic fragile; bare `try/except` swallows failures |

Model files stored as bare filenames in CWD (e.g., `yolo11n.pt`). No `models/` directory.

---

## Tracking — `src/tracking/tracker.py`

**File**: `src/tracking/tracker.py` (267 lines)

### `Track` dataclass (line 40)
```python
id: int                    # boxmot track_id
class_id: int
class_name: str
confidence: float
bbox: tuple[int,int,int,int]
age: int = 0
embedding: np.ndarray | None = None  # OSNet ReID feature vector
embedding_frame: int = -1
camera_id: str = ""        # which camera produced this track
global_id: int = -1        # placeholder for identity layer (Phase 4.2)
attributes: dict = field(default_factory=dict)
```

`to_dict()` (line 52): serializes all fields except `embedding` (stored separately in ObjectHistory as base64).

### `Tracker` class (line 143)

| Method | Lines | Purpose |
|--------|-------|---------|
| `__init__()` | 144-190 | Options: `track_thresh`, `track_buffer`, `match_thresh`, `use_reid`, `reid_model`, `device`, `use_cmc`, `camera_id` |
| `_compute_embedding()` | 188-205 | Crops bbox from frame, runs `ReID.get_features()`, returns flattened numpy |
| `_bbox_iou()` | 207-214 | Static IoU calculator used for track→detection matching |
| `update()` | 216-267 | Main: detections → Track list |

### ReID Architecture
- **BoT-SORT** chosen when `use_reid=True` AND `BotSort` importable (line 163)
- **ByteTrack** fallback when `use_reid=False` (line 180)
- `with_reid=True` (line 174) — BoT-SORT uses appearance + IoU for matching
- Embeddings computed ONCE per detection (line 226), passed to boxmot via `embs=` kwarg (line 237)
- **⚠️ boxmot >=21 required** for `reid_model=` and `use_cmc=` params; `requirements.txt` says `>=19` → wrong
- ReID model weights: `osnet_x1_0_msmt17.pt` (default), `osnet_ain_x1_0_msmt17.pt` (alt). Downloaded automatically by boxmot.

### Track Init Flow (lines 216-267)
1. Build `dets_np` from detections
2. Compute `det_embs` for all detections
3. `self.tracker.update(dets_np, frame, embs=embs_np)` — boxmot skips internal ReID
4. For each output track: match to detection by IoU > 0.3, reuse detection embedding for refresh
5. Fallback: `_compute_embedding` only if IoU match fails
6. Stale track cleanup (lines 274-278)

### `TrackStateMachine` — **DEAD CODE**
**File**: `src/tracking/state.py` (66 lines)
Full FSM with transitions (NEW→ACTIVE→OCCLUDED→LOST→MERGED→ENDED). Never imported anywhere.

### Embedding flow through the system
```
Tracker._compute_embedding() → Track.embedding → ObjectHistory (copied) → export → embedding_b64 (base64) → merge_fragments (pass-through) → pipeline output → DB
```

---

## Pipeline — `src/pipeline.py`

**File**: `src/pipeline.py` (469 lines)
**Function**: `analyze_video()` — the central orchestrator

### Parameters (line 33-66)
`video_path`, `output_dir`, `model_family`, `model_size`, `conf_threshold`, `device`, `max_frames`, `track_thresh`, `match_thresh`, `track_low_thresh`, `track_buffer`, `trail_length`, `use_reid`, `reid_model`, `zone_config`, `calibration_config`, `capture_evidence`, `filter_stationary_objects`, `min_move_distance`, `target_classes`, `use_tensorrt`, `plate_read_interval`, `use_cmc`, `reid_refresh_interval`, `reid_new_track_frames`, `min_face_size`, `face_interval`, `skip_face`, `camera_id`, `detector`

### Frame loop (lines 148-289) — per-frame execution order
1. `detector.detect(frame)` → detections
2. `tracker.update(detections, frame)` → tracks
3. `history.update(tracks)` → ObjectHistory
4. Per-track: zone/gate, dwell, abandoned, interaction
5. `vehicle_analyzer.process_frame()` → vehicle events
6. `scene_analyzer.process_frame()` → carrying/overload events
7. `identity_tracker.update()` per track
8. `predictor.update()` per track
9. `face_recognizer.process_frame()` → face events (if not skip_face)
10. `correlator.process_event()` → incidents
11. Annotator render → video output

### Post-loop (lines 293-370)
1. `history.export()` → objects
2. `merge_fragments()` (if `use_reid`) — joins fragments by class + spatial/temporal proximity
3. `filter_stationary()` — removes non-moving objects
4. `movement_stats()` per object + calibrator world-coordinate conversion
5. **Database**: `AnalyticsDB` → start_run, insert_tracks_batch, insert_events_batch, upsert_gate_count, finish_run
6. JSON report → `analytics.json`
7. Text summary → `summary.txt`

### Output dict structure (line 331)
```python
{
    "video", "video_duration_sec", "total_frames_processed", "fps", "resolution",
    "total_objects_tracked", "total_detections", "object_counts",
    "objects": [{"id", "class", "class_id", "first_frame", "last_frame", "duration_frames",
                  "path", "confidence", "movement": {...},
                  "camera_id", "global_id", "embedding_b64"}],
    "output_video", "zones", "calibration", "gate_counts", "dwell_summary",
    "events": [...],
    "vehicles": {...}, "vehicle_list": [...],
    "scene_events": {...},
    "identities": [{"track_id", "name", "confidence"}],
    "incidents": [...],
    "evidence_clips": [...] (optional)
}
```

---

## Scene Intelligence Modules — `src/analytics/`

### `object_history.py` (90 lines)
**Class**: `ObjectHistory`
| Method | Purpose |
|--------|---------|
| `update(tracks, frame)` | Creates/appends `ObjectRecord` per track |
| `export()` | Returns `list[dict]` with id, class, path, camera_id, global_id, embedding_b64 |

`ObjectRecord` dataclass (line 12): `id`, `class_name`, `class_id`, `first_frame`, `last_frame`, `positions[(cx,cy)]`, `bboxes`, `camera_id`, `global_id`, `embedding`

### `merge_fragments.py` (95 lines)
**Function**: `merge_fragments(objects, track_buffer=300, max_center_distance=100.0)`
Merges track fragments by `class_name` + time gap (`track_buffer`) + centroid distance (`max_center_distance`). ⚠️ Does NOT use embedding similarity for merge decisions — only spatial proximity. Propagates `camera_id`, `global_id`, `embedding_b64`.

### `zones.py`
**Classes**: `Zone` (polygon), `LineGate`, `ZoneManager`
**Query**: `zones_at(x, y)` → zones containing point. `check_gate_crossing(track_id, cx, cy)` → gate events.

### `counting.py`
**Class**: `GateCounter` — `record(gate, track_id, direction)` → entries/exits/net per gate.

### `dwell.py`
**Class**: `DwellTracker` — per `(track_id, zone)` duration tracking with `update()` + `current_dwell()`.

### `events.py` — **NOT `src/models/event.py`**
**Class**: `EventDetector` — factory for `check_zone_entry`, `check_zone_exit`, `check_loitering`.

### `abandoned.py`
**Class**: `AbandonedDetector` — marks non-person tracks as "abandoned" if stationary beyond threshold.

### `movement.py`
**Function**: `movement_stats(path)` → `{distance_pixels, average_speed, direction}` (8 compass + stationary).

### `interaction.py`
**Class**: `InteractionModel` — proximity-based person-person / person-object interaction detection.

### `stationary_filter.py`
**Function**: `filter_stationary(objects, min_path_distance, min_duration)` — removes tracks below distance threshold.

### `calibration.py`
**Class**: `Calibrator` — 4-point homography via `cv2.findHomography`. `image_to_world()`, `path_length_in_world()`, `speed_in_world()`.

### `identity.py`
**Class**: `IdentityConfidence` — per-track stats: stability, appearance variance, occlusion count.

### `prediction.py`
**Class**: `TrackPredictor` — 4-state Kalman filter per track (`update()` → `predict()` → `predicted_position()`).

### `time_sync.py`
**Class**: `TimeSync` — frame index → UTC/camera/processing timestamps.

### `correlation.py` (130 lines)
**Class**: `EventCorrelator` — 4 incident rules:
1. `suspicious_activity` (critical): zone entry + 2 of {loitering, abandoned, fast movement}
2. `prolonged_loitering` (high): loitering + interaction
3. `gate_breach` (high): gate crossing + optional fast movement
4. `object_drop` (high): abandoned + interaction
Single-camera only. No cross-camera correlation.

### `evidence.py`
**Class**: `EvidenceCapture` — sliding window frame buffer, saves mp4 clips on event trigger. Configurable pre/post frames.

---

## Scene Understanding — `src/analytics/scene/`

### `orchestrator.py`
Creates `CarryingDetector` + `OverloadedVehicleDetector`, calls both per frame.

### `carrying.py`
Person carrying detection: IoU overlap between person and carryable object (backpack, handbag, suitcase, laptop, cell phone, book, bottle, cup, umbrella, knife, baseball bat) + co-motion direction similarity (dot product of velocity vectors for ≥N frames).

### `overloaded_vehicle.py`
Detects vehicles with ≥N persons co-moving inside expanded vehicle bbox. Uses same co-motion analysis.

---

## Vehicle Intelligence — `src/analytics/vehicle/` (13 files)

### `orchestrator.py`
Main controller. Called every frame by pipeline. Manages per-track attribute state machine and plate detection pipeline:
1. Skip if attributes already LOCKED
2. Quality check frame
3. `PlateDetector.detect()` → plate crops
4. `PlateReader.read_async()` (OCR in thread pool)
5. Update `TopKBuffer` with candidates
6. When confident: extract color + size, compute speed, register vehicle

### `plate_detector.py`
PaddleOCR's `OCR.ocr()` for plate region detection. Contour-based fallback (aspect ratio filtering).

### `plate_reader.py`
OCR text extraction: PaddleOCR primary, EasyOCR fallback. Preprocessing (CLAHE, sharpening, upscale). Post-processing (validation, cleaning).

### `ocr_pool.py`
Singleton pattern: one `PaddleOCR` instance shared via `get_ocr_pool()` → `ThreadPoolExecutor`.

### `attributes.py`
Vehicle color: K-means dominant color extraction from lower-portion crop.
Vehicle size: classified by bbox dimensions.

### `attribute_cache.py`
State machine per track: `UNKNOWN → PROCESSING → LOCKED | FAILED`.

### `candidate_buffer.py`
Top-K `PlateCandidate` buffer with quality scoring: sharpness (Laplacian variance), contrast, viewing angle, plate pixel size.

### `registry.py`
`VehicleRegistry` — plate-keyed records. Each record: first/last seen, visit count, color, type, parking duration.

### `validation.py`
Plate text cleaning: strip non-alphanumeric, country-pattern validation (US, UK, EU, generic).

### `frame_quality.py`
Quality metrics: sharpness (Laplacian), plate size ratio, viewing angle deviation, brightness/contrast, motion blur.

### `preprocessing.py`
Image enhancement: bicubic upscale, sharpening kernel, CLAHE.

### `events.py`
Event constructors: `speeding()`, `parking_event()`, `plate_read()`, `repeat_visitor()`.

---

## Face Recognition — `src/analytics/face_recognition.py`

**File**: `src/analytics/face_recognition.py` (451 lines)

### `FaceGallery` (line 25)
Persistent store of known faces via JSON file (`face_gallery.json`). Embeddings base64-encoded.
Methods: `match(embedding, threshold)` → `(name, similarity)`, `add(name, embedding)`, `known_names()`, `remove()`, `save()`, `load()`.

### `FaceRecognizer` (line 103)
Uses insightface `buffalo_l` (RetinaFace+ArcFace).
Methods:
- `process_frame(frame, tracks, frame_idx)` — async via `ThreadPoolExecutor`: crops person bbox upper-portion, detects face, extracts embedding, matches gallery. Returns events on `_confirm_frames` agreement.
- `add_known_face(name, image)` — register from image.
- `get_all_identities()` → `{track_id: (name, confidence)}`

Key defaults: `min_face_size=40`, `match_threshold=0.45`, `confirm_frames=5`, `face_interval=6`

### `UnknownFaceStore` (line 348)
Captures unrecognized faces as images + embeddings to `unknown_faces/`. Supports batch matching and promotion to gallery.

---

## Multi-Camera — `src/optimization/multi_stream.py`

**File**: `src/optimization/multi_stream.py` (146 lines)

**Function**: `process_cameras(camera_configs, output_dir, mosaic_layout, max_workers)`
**Class**: `MultiCameraPipeline` (thin wrapper, line 132)

### Architecture
- **`ThreadPoolExecutor(max_workers=4)`** — 1 shared `YOLODetector` (singleton), N independent trackers/analytics/events
- Each camera: `_run_one(i, cfg)` → `analyze_video(video_path=cfg["video_path"], detector=shared, camera_id=..., **pipe_kwargs)`
- Config keys: `video_path`, `name`, `model_family`, `model_size`, `device`, `target_classes`, `use_tensorrt`, `camera_id`, etc.
- Output: `output_dir/camera_N/analytics.json`, `output_dir/multi_camera_report.json`

### ⚠️ Cross-camera identity: **NOT IMPLEMENTED**
- `global_id` on Track (default -1) never assigned
- No `IdentityManager`, no embedding database, no cross-camera ReID matching
- `Camera.topology` field exists but never consumed
- Mosaic is visual-only (2×2/horizontal/vertical via `src/visualization/mosaic.py`)

---

## Output — `src/visualization/`

### `annotator.py`
**Class**: `Annotator` — draws tracks (trail lines, bboxes, IDs), zone overlays, gate markers, event ticker, face labels. Outputs to `output_tracking.mp4` via `cv2.VideoWriter` (mp4v codec).

### `mosaic.py`
**Function**: `create_mosaic(video_paths: dict[str, str], output_path, layout="2x2")` — N-up sync grid. `2x2` (exactly 4), `horizontal`, `vertical`, or auto-grid.

### `zone_renderer.py`
`draw_zones()`, `draw_gates()`, `draw_event_ticker()` — called by Annotator.

---

## Database — `src/db/`

### `schema.py` (6 tables)
| Table | Columns |
|-------|---------|
| `cameras` | id, name, source, fps, resolution, location, gps, timezone, status |
| `analysis_runs` | id, camera_id, video_path, model_family, model_size, total_frames, fps, resolution, calibration, zones_config |
| `tracks` | id, run_id, camera_id, track_id, class_name, class_id, first/last_frame, duration, distance, speed, direction, path_json, embedding_b64 |
| `events` | id, run_id, camera_id, event_type, track_id, zone, severity, location_x/y, message, utc_time, frame_index |
| `incidents` | id, run_id, incident_type, severity, summary, track_ids_json, events_json |
| `gate_counts` | id, run_id, camera_id, gate_name, entries, exits, net |
| `evidence_clips` | id, run_id, camera_id, event_type, track_id, clip_path |

### `repository.py` (172 lines)
**Class**: `AnalyticsDB` — SQLite3 wrapper. CRUD for all tables. Batch inserts. Query helpers.
**Wired into pipeline**: Yes (Phase 4.1). Calls at end of `analyze_video()`.
Path: `{output_dir}/analytics.db`

---

## Configuration — `src/config.py` + `configs/`

### `configs/defaults/` (YAML, loaded by `src/config.py`)

| File | Keys |
|------|------|
| `detector.yaml` | `model_family`, `model_size`, `conf_threshold`, `device`, `use_tensorrt`, `tensorrt_half`, `target_classes` |
| `tracker.yaml` | `use_reid`, `reid_model`, `track_high_thresh`, `track_low_thresh`, `track_buffer`, `match_thresh`, `trail_length` |
| `analytics.yaml` | `zones_config_path`, `calibration_config_path`, `capture_evidence`, `filter_stationary_objects`, `min_move_distance`, `loitering`, `interaction`, `abandoned`, `evidence` |
| `camera.yaml` | `fps: 25`, `resolution: [640, 360]`, `timezone: UTC` |
| `vehicle.yaml` | `enabled: true`, `parking_timeout_sec: 300`, `speeding_threshold_mps: 11.0`, `plate_reader`, `attributes` |

### `configs/*.json` (runtime configs)
- `demo_zones.json` — zone polygons + gate lines
- `demo_calibration.json` — 4-point homography
- `cameras.json` — camera definitions with topology
- `4cameras.json` — 4-camera test config

---

## Event Data Model — `src/models/event.py`

**Class**: `Event` (dataclass): `event_type`, `track_id`, `class_name`, `zone`, `duration`, `location[x,y]`, `message`, `severity`, `confidence`. Auto-severity map in `SEVERITY_MAP`.

**Class**: `EventStore` — in-memory list. `add()`, `by_type()`, `by_severity()`, `critical()`, `recent(N)`, `export()`, `save()`.

---

## Plugin System — `src/plugin/base.py` (STUB)
Abstract base class. No plugins registered or loaded anywhere.

---

## Key Data Flow Summary

```
VideoLoader (cv2)
  → YOLODetector.detect() → list[Detection]
    → Tracker.update() → list[Track]  (BoT-SORT + ReID, with_reid=True)
      → ObjectHistory.update()
      → Per-frame analytics (zones, dwell, abandoned, interaction, vehicle, scene, face, correlation)
      → Annotator (video output)
      → [end of frame loop]
    → ObjectHistory.export() → list[dict]
    → merge_fragments() (if use_reid)
    → filter_stationary()
    → movement_stats() per object
    → AnalyticsDB (tracks, events, gate_counts)
    → JSON report (analytics.json)
    → Text summary (summary.txt)
```

---

## Identity Layer — `src/identity/` (Phase 4.2, implemented)

Cross-camera identity matching — assigns persistent `global_id`s to
observations across cameras using ReID embeddings.

| File | Responsibility |
|------|----------------|
| `identity_manager.py` | `IdentityManager` — ingest(camera_objects) → match → assign global_id → write report. `load_topology()` parses camera adjacency configs |
| `embedding_store.py` | `IdentityStore` — JSON persistence (base64 embeddings, up to 8 per identity), thread-safe, `create()` / `add_embedding_and_observation()` / `embeddings()` |
| `matcher.py` | Cosine similarity matching. Thresholds: `>=0.85` confident, `>=0.70` uncertain (still assigned, flagged), `<0.70` new identity. Topology boost (+0.03, cap 0.80) for candidates seen on adjacent cameras |
| `global_identity.py` | `GlobalIdentity` (id, class, embeddings, observations) + `Observation` (camera_id, local_track_id, frames) + `to_dict()` |

### Integration
- Called by `process_cameras()` after ALL cameras finish (multi_stream.py):
  loads each camera's `analytics.json` → `IdentityManager.ingest()` →
  writes `output_dir/global_identity_report.json` → report includes
  `global_identities` summary.
- Mutates each object dict: sets `global_id` + `identity_match_score`.
- Store persists at `output_dir/identity_store.json` — subsequent runs
  match against previously learned identities (cross-run identity).
- Single-camera runs keep `global_id=-1` (no ingestion happens).

### Timeline output
`report["timelines"][gid]` = [{camera_id, local_track_id, first_frame,
last_frame, match_score}] sorted by frame — the "where has this person
been" path reconstruction.

## Identity Layer — NOT IMPLEMENTED (Phase 4.3)

- Cross-camera EVENT correlation (Person 42 entered Gate A → Hallway → Restricted)
- Dashboard / search UI (Phase 5)

---

## Classification of every module

| Module | Status | Lines | Key Class/Function |
|--------|--------|-------|--------------------|
| `detection/yolo_detector.py` | ✅ Full | 230 | `YOLODetector` |
| `optimization/tensorrt_export.py` | ✅ Full | 73 | `export_to_engine()` |
| `tracking/tracker.py` | ✅ Full | 267 | `Tracker`, `Track` |
| `tracking/state.py` | ✅ Wired (was dead) | 66 | `TrackStateMachine` — used by Tracker for occlusion/ended stats |
| `pipeline.py` | ✅ Full | 469 | `analyze_video()` |
| `analytics/object_history.py` | ✅ Full | 90 | `ObjectHistory`, `ObjectRecord` |
| `analytics/merge_fragments.py` | ✅ Full | 95 | `merge_fragments()` |
| `analytics/zones.py` | ✅ Full | — | `ZoneManager` |
| `analytics/counting.py` | ✅ Full | — | `GateCounter` |
| `analytics/dwell.py` | ✅ Full | — | `DwellTracker` |
| `analytics/events.py` | ✅ Full | — | `EventDetector` |
| `analytics/abandoned.py` | ✅ Full | — | `AbandonedDetector` |
| `analytics/movement.py` | ✅ Full | — | `movement_stats()` |
| `analytics/interaction.py` | ✅ Full | — | `InteractionModel` |
| `analytics/stationary_filter.py` | ✅ Full | — | `filter_stationary()` |
| `analytics/calibration.py` | ✅ Full | — | `Calibrator` |
| `analytics/identity.py` | ✅ Full | — | `IdentityConfidence` |
| `analytics/prediction.py` | ✅ Full | — | `TrackPredictor` |
| `analytics/time_sync.py` | ✅ Full | — | `TimeSync` |
| `analytics/correlation.py` | ✅ Full | 130 | `EventCorrelator` (4 rules) |
| `analytics/evidence.py` | ✅ Full | — | `EvidenceCapture` |
| `analytics/face_recognition.py` | ✅ Full | 451 | `FaceRecognizer`, `FaceGallery` |
| `analytics/scene/orchestrator.py` | ✅ Full | — | `SceneAnalyzer` |
| `analytics/scene/carrying.py` | ✅ Full | — | `CarryingDetector` |
| `analytics/scene/overloaded_vehicle.py` | ✅ Full | — | `OverloadedVehicleDetector` |
| `analytics/vehicle/orchestrator.py` | ✅ Full | — | `VehicleAnalyzer` |
| `analytics/vehicle/plate_detector.py` | ✅ Full | — | `PlateDetector` |
| `analytics/vehicle/plate_reader.py` | ✅ Full | — | `PlateReader` |
| `analytics/vehicle/ocr_pool.py` | ✅ Full | — | `get_ocr_pool()` |
| `analytics/vehicle/attributes.py` | ✅ Full | — | Color + size extraction |
| `analytics/vehicle/attribute_cache.py` | ✅ Full | — | State machine |
| `analytics/vehicle/candidate_buffer.py` | ✅ Full | — | TopK buffer |
| `analytics/vehicle/registry.py` | ✅ Full | — | `VehicleRegistry` |
| `analytics/vehicle/validation.py` | ✅ Full | — | Plate validation |
| `analytics/vehicle/frame_quality.py` | ✅ Full | — | Quality scoring |
| `analytics/vehicle/preprocessing.py` | ✅ Full | — | Image enhancement |
| `analytics/vehicle/events.py` | ✅ Full | — | Event constructors |
| `optimization/multi_stream.py` | ✅ Full | 146 | `MultiCameraPipeline` + global identity ingest |
| `optimization/profiler.py` | ✅ Full | — | `PipelineProfiler` (not wired to pipeline) |
| `identity/identity_manager.py` | ✅ Full (Phase 4.2) | — | `IdentityManager` |
| `identity/embedding_store.py` | ✅ Full (Phase 4.2) | — | `IdentityStore` |
| `identity/matcher.py` | ✅ Full (Phase 4.2) | — | `find_best_match()` |
| `identity/global_identity.py` | ✅ Full (Phase 4.2) | — | `GlobalIdentity`, `Observation` |
| `db/schema.py` | ✅ Full | — | 6 tables, SQLite |
| `db/repository.py` | ✅ Full | 172 | `AnalyticsDB` (wired) |
| `models/event.py` | ✅ Full | — | `Event`, `EventStore` |
| `models/camera.py` | ✅ Full | — | `Camera` |
| `models/zone.py` | ✅ Full | — | `Zone`, `LineGate` |
| `events/bus.py` | ⚠️ Partial | — | Event bus (not wired) |
| `plugin/base.py` | 📦 Stub | — | Abstract base only |
| `visualization/annotator.py` | ✅ Full | — | `Annotator` |
| `visualization/mosaic.py` | ✅ Full | — | `create_mosaic()` |
| `visualization/zone_renderer.py` | ✅ Full | — | Drawing helpers |

---

## Known Issues

1. **boxmot version**: `requirements.txt` says `>=19.0.0`, needs `>=21.0.0` for `reid_model=` and `use_cmc=` params
2. **Engine filename**: FP16 and FP32 both write to `yolo11n.engine` — `half` not encoded in path
3. **Batch inference unused**: `self.batch_size` stored but `detect()` called per frame in pipeline
4. **detect/detect_batch duplication**: ~25 lines of identical filtering logic
5. **bench_realtime fidelity**: Missing merge, zone logic, scene analysis, video encoding vs production
6. **Cross-camera EVENT correlation**: identity layer exists; event-level correlation across cameras not yet implemented
7. **Event bus**: `src/events/bus.py` exists but not wired
9. **Profiler**: Not wired into production pipeline
10. **Cache key omissions**: `target_classes`, `tensorrt_half`, `batch_size` not in shared detector cache key
