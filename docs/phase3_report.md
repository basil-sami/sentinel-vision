# Sentinel Vision — Phase 3 Report

## Objective
Build a Scene Intelligence Layer that transforms tracked objects into meaningful security analytics — reasoning over position, movement, time-in-zone, and behavioral patterns.

## Architecture

### Data Flow
```
Video → YOLO11x → BoT-SORT + ReID → ObjectHistory
                                         ↓
                              ZoneManager → entry/exit events
                              GateCounter → counting per virtual line
                              DwellTracker → time-in-zone per object
                              EventDetector → loitering rules
                              AbandonedDetector → owner-separation logic
                              Movement → direction, speed, distance
                                         ↓
                              EventStore + Video overlay + JSON/Summary
```

### Design Principle
All intelligence operates on **tracked objects**, not individual frames. The tracker assigns persistent IDs; Phase 3 reasons over their trajectories, zone interactions, and temporal patterns.

---

## Feature 1 — Zone System

**Files:** `src/models/zone.py`, `src/analytics/zones.py`

Configurable polygon regions of interest with:
- Point-in-polygon containment test (ray-casting algorithm)
- Per-zone type coloring (`restricted`, `entrance`, `parking`, `walkway`)
- Zone entry/exit detection per tracked object
- Active zone highlighting (yellow glow when object inside)

**Example config:**
```python
{
    "zones": [
        {"name": "Main Gate", "type": "restricted",
         "polygon": [[100,100], [300,100], [300,300], [100,300]]}
    ],
    "gates": [
        {"name": "North Gate", "p1": [0,200], "p2": [640,200]}
    ]
}
```

---

## Feature 2 — Entry/Exit Counting

**Files:** `src/analytics/counting.py`, `src/models/zone.py` (LineGate)

Virtual gate counting via line-crossing detection:
- Define a line between two points
- Detect direction of crossing using sign-change of half-plane function
- Deduplicate counts per track + direction pair
- Output: entries, exits, net count per gate

**Example output:**
```json
{
    "North Gate": {"entries": 25, "exits": 18, "net": 7}
}
```

---

## Feature 3 — Dwell Time Analysis

**File:** `src/analytics/dwell.py`

Tracks time spent inside zones:
- Frame-accurate entry/exit timestamps
- Current dwell (real-time during processing)
- Total, average, and max dwell durations per zone
- Unique object count per zone

**Example output:**
```json
{
    "Main Gate": {
        "total_dwell_frames": 4500,
        "average_dwell_frames": 150.0,
        "max_dwell_frames": 600,
        "unique_objects": 30
    }
}
```

---

## Feature 4 — Loitering Detection

**File:** `src/analytics/events.py`

Rule-based detection configurable by class:
- Person threshold: 600s (10 min)
- Vehicle threshold: 300s (5 min)
- Generates `possible_loitering` events with duration and location
- One warning per threshold-bucket to avoid spam

**Example event:**
```json
{
    "type": "possible_loitering",
    "track_id": 31,
    "class": "person",
    "zone": "Main Gate",
    "duration": 650.0,
    "location": [320, 240]
}
```

Thresholds editable in `src/analytics/events.py` `loiter_config` dict.

---

## Feature 5 — Abandoned Object Detection

**File:** `src/analytics/abandoned.py`

Behavior-based detection (not simple bag classification):
1. Track all objects (people + vehicles + objects)
2. Associate non-person objects to nearest person within 80px
3. When owner (person) moves away, mark object as stationary
4. After stationary threshold (`track_buffer` frames, default 300), generate alert
5. Includes last-known owner ID in the event

**Example event:**
```json
{
    "type": "abandoned_object",
    "track_id": 55,
    "class": "backpack",
    "duration": 12.0,
    "location": [400, 180],
    "message": "Abandoned backpack ID 55 (owner ID 10)"
}
```

---

## Feature 6 — Movement Analytics

**File:** `src/analytics/movement.py`

Per-track movement statistics:
- **Distance:** total Euclidean distance traveled (pixels)
- **Average speed:** distance / number of frames (pixels/frame)
- **Direction:** 8-point compass (N, NE, E, SE, S, SW, W, NW) from start-to-end vector

**Example output:**
```json
{
    "track_id": 17,
    "distance_pixels": 4520.0,
    "average_speed": 3.2,
    "direction": "north"
}
```

---

## Visualization

**File:** `src/visualization/zone_renderer.py`

Enhanced video output:
- **Zone overlays:** semi-transparent colored fills + borders
- **Active zone glow:** zones highlight yellow when objects are inside
- **Zone labels:** name + type badge centered on polygon
- **Gate lines:** yellow dashed lines with labels
- **Gate counter HUD:** live entry/exit/net counts in top-left
- **Event ticker:** last 5 events displayed at bottom of frame

---

## Pipeline Integration

**File:** `src/pipeline.py`

The main loop now runs per frame:
1. Detect → Track → Update history
2. Zone entry/exit detection → fire events
3. Dwell tracking → fire loitering events
4. Gate crossing detection → update counters
5. Abandoned object detection → fire events
6. Render zones, gates, event ticker, counters on frame

Post-processing adds movement stats to each object and includes all zone/count/event data in JSON output.

---

## Output Structure

**analytics.json** now includes:
```json
{
    "video": "...",
    "total_objects_tracked": 83,
    "zones": { "config": [...] },
    "gate_counts": { "North Gate": {"entries": 25, "exits": 18} },
    "dwell_summary": { "Main Gate": {"total_dwell_frames": 4500} },
    "events": [ {"type": "zone_entry", ...}, ... ],
    "objects": [ {"id": 1, "movement": {...}, ...}, ... ]
}
```

**summary.txt** now includes:
- Zone configurations
- Gate count breakdown
- Last 10 events
- Per-object movement stats

---

## Files Created

| File | Purpose |
|---|---|
| `src/models/zone.py` | Zone + LineGate data models |
| `src/models/event.py` | Event dataclass + EventStore |
| `src/analytics/zones.py` | ZoneManager with polygon/gate logic |
| `src/analytics/counting.py` | GateCounter with dedup |
| `src/analytics/dwell.py` | DwellTracker for time-in-zone |
| `src/analytics/events.py` | EventDetector with loitering rules |
| `src/analytics/abandoned.py` | AbandonedDetector with owner-separation |
| `src/analytics/movement.py` | Direction, speed, distance |
| `src/visualization/zone_renderer.py` | Zone/gate/counter/event video overlay |
| `configs/demo_zones.json` | Example zone configuration |
| `docs/phase2_report.md` | Phase 2 retrospective |
| `docs/phase3_report.md` | This document |

## Files Modified

| File | Change |
|---|---|
| `src/analytics/object_history.py` | Store per-frame bbox data |
| `src/pipeline.py` | Zone/event/abandoned integration loop |
| `notebooks/sentinel_demo.ipynb` | Phase 3 branch, zone config, all params |

## Next Steps (Phase 4 candidates)
- License plate recognition
- Face detection / recognition
- Multi-camera identity linking
- Real-time alert webhooks
- Dashboard UI
