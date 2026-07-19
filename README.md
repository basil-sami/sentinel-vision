# Sentinel Vision

A proof-of-concept framework for automated video surveillance analytics using modern computer vision.

## Overview

Sentinel Vision processes unseen CCTV-style video footage and extracts meaningful information about objects, movement, and events.

**Capabilities:**
- Object detection (YOLO11x)
- Multi-object tracking with persistent IDs (BoT-SORT + ReID)
- Object trajectory analysis
- Configurable zone monitoring (entry/exit, restricted areas)
- Virtual gate counting with direction
- Dwell time analysis
- Loitering detection (configurable per class)
- Abandoned object detection (owner-separation logic)
- Movement analytics (direction, speed, distance)
- Event logging and alert generation
- Annotated video output with zone overlays
- JSON analytics report
- Text summary report

## Project Structure

```
sentinel-vision/
├── README.md
├── requirements.txt
├── configs/             # Zone configuration files
├── docs/                # Phase reports
├── src/
│   ├── video/           # Video loading and frame extraction
│   ├── detection/       # Object detection (YOLO)
│   ├── tracking/        # Multi-object tracking (BoT-SORT + ReID)
│   ├── analytics/       # Object stats, zones, events, dwell, abandoned, movement
│   ├── models/          # Zone, Event data models
│   └── visualization/   # Annotator, zone renderer
├── notebooks/           # Colab-compatible notebooks
├── outputs/             # Generated videos and reports
├── tests/               # Unit tests
└── models/              # Downloaded model weights
```

## Quick Start

```python
from src.pipeline import analyze_video
import json

zone_config = json.load(open("configs/demo_zones.json"))
analyze_video("input_video.mp4", zone_config=zone_config)
```

Output:
- `outputs/output_tracking.mp4` — Annotated video with zones, counters, events
- `outputs/analytics.json` — Structured JSON report
- `outputs/summary.txt` — Human-readable summary

## Phases

- **Phase 1:** Video loading, YOLO detection, bounding box visualization
- **Phase 2:** Multi-object tracking with persistent IDs (BoT-SORT + ReID)
- **Phase 3:** Scene intelligence — zones, counting, dwell, loitering, abandoned objects, movement analytics

