# Sentinel Vision

A proof-of-concept framework for automated video surveillance analytics using modern computer vision.

## Overview

Sentinel Vision processes unseen CCTV-style video footage and extracts meaningful information about objects, movement, and events.

**Capabilities:**
- Object detection (YOLO-based)
- Multi-object tracking with persistent IDs
- Object trajectory analysis
- Basic event detection (loitering, line crossing, crowd detection)
- Annotated video output
- JSON analytics report
- Text summary report

## Project Structure

```
sentinel-vision/
├── README.md
├── requirements.txt
├── src/
│   ├── video/          # Video loading and frame extraction
│   ├── detection/      # Object detection (YOLO)
│   ├── tracking/       # Multi-object tracking (ByteTrack)
│   ├── analytics/      # Object statistics and event detection
│   └── visualization/  # Bounding boxes, trails, annotated video
├── notebooks/          # Colab-compatible notebooks
├── outputs/            # Generated videos and reports
├── tests/              # Unit tests
├── models/             # Downloaded model weights
└── docs/               # Architecture decisions
```

## Quick Start

```python
from src.pipeline import analyze_video

analyze_video("input_video.mp4")
```

Output:
- `outputs/output_tracking.mp4` — Annotated video
- `outputs/analytics.json` — Structured JSON report
- `outputs/summary.txt` — Human-readable summary

## Phases

- **Phase 1:** Video loading, YOLO detection, bounding box visualization
- **Phase 2:** Multi-object tracking with persistent IDs
- **Phase 3:** Movement trajectories, statistics, JSON reports
- **Phase 4:** Event detection (line crossing, dwell time, counting)
