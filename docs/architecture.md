# Architecture Decisions

## Design Principles

- **Modular**: Each concern (detection, tracking, analytics, visualization) lives in its own module.
- **Swappable models**: Detectors and trackers implement a common interface so backends can be replaced (e.g., YOLO → RT-DETR, ByteTrack → BoT-SORT).
- **Colab-first**: All dependencies are pip-installable; no system-level modifications required.
- **Git as memory**: Every session starts by reading the repo; no assumptions about previous state.

## Component Overview

### Video Loader (`src/video/loader.py`)
- Wraps OpenCV `VideoCapture`
- Provides frame iteration, seeking, metadata (fps, dimensions, count)
- Outputs RGB numpy arrays

### Detection (`src/detection/yolo_detector.py`)
- Uses Ultralytics YOLO (v11) pretrained on COCO
- Filters to surveillance-relevant classes: person, bicycle, car, motorcycle, bus, truck
- Returns list of `Detection` dataclass instances

### Visualization (`src/visualization/annotator.py`)
- Draws bounding boxes and labels per-detection
- Writes annotated frames to mp4 via OpenCV `VideoWriter`
- Color assignment deterministic by object ID

### Pipeline (`src/pipeline.py`)
- Orchestrates loader → detector → annotator
- Collects per-frame detection stats
- Writes `analytics.json` and `summary.txt`

## Output Formats

### JSON (`outputs/analytics.json`)
Top-level fields: `video`, `video_duration_sec`, `fps`, `resolution`, `total_detections`, `object_counts`, per-frame `detections` array.

### Summary (`outputs/summary.txt`)
Human-readable plaintext with video info and object tallies.

## Future Phases

- **Phase 2**: ByteTrack integration for persistent object IDs across frames
- **Phase 3**: Trajectory extraction, dwell time, heatmaps
- **Phase 4**: Rule-based events (line crossing, loitering, crowd detection)
