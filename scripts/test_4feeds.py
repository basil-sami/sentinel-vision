"""Multi-feed test: run analytics on 4 camera feeds (1 original + 3 combos)."""

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from src.pipeline import analyze_video

TEST_VIDEOS = {
    "cam00_original": "test_videos/cam00_original.mp4",
    "cam01_mirror_slow": "test_videos/cam01_mirror_slow.mp4",
    "cam02_mirror_reverse": "test_videos/cam02_mirror_reverse.mp4",
    "cam03_slow_reverse": "test_videos/cam03_slow_reverse.mp4",
}

OUTPUT_BASE = SRC_DIR / "outputs" / "4feeds"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

ZONE_CONFIG = SRC_DIR / "configs" / "demo_zones.json"
CALIB_CONFIG = SRC_DIR / "configs" / "demo_calibration.json"

ALL_RESULTS: dict[str, dict] = {}

for cam_name, rel_path in TEST_VIDEOS.items():
    video_path = str(SRC_DIR / rel_path)
    cam_output = str(OUTPUT_BASE / cam_name)

    print(f"\n{'='*60}")
    print(f"Processing: {cam_name} ({rel_path})")
    print(f"{'='*60}")

    zones = json.loads(ZONE_CONFIG.read_text())
    calib = {}
    if CALIB_CONFIG.exists():
        calib = json.loads(CALIB_CONFIG.read_text())

    result = analyze_video(
        video_path=video_path,
        output_dir=cam_output,
        model_family="yolo11",
        model_size="nano",
        conf_threshold=0.4,
        device="cpu",
        max_frames=200,
        track_thresh=0.4,
        match_thresh=0.7,
        track_low_thresh=0.1,
        track_buffer=450,
        use_reid=True,
        reid_model="x1_0",
        zone_config=zones,
        calibration_config=calib,
        capture_evidence=True,
        filter_stationary_objects=True,
        min_move_distance=20.0,
    )
    ALL_RESULTS[cam_name] = result
    print(f"  Tracks: {result['total_objects_tracked']}, "
          f"Detections: {result['total_detections']}, "
          f"Events: {len(result['events'])}")

# --- Compare across cameras ---
print(f"\n\n{'='*60}")
print("CROSS-CAMERA COMPARISON")
print(f"{'='*60}")

headers = ["Metric"] + list(TEST_VIDEOS.keys())
rows: list[list[str]] = []
rows.append(["Tracks"] + [str(r["total_objects_tracked"]) for r in ALL_RESULTS.values()])
rows.append(["Detections"] + [str(r["total_detections"]) for r in ALL_RESULTS.values()])
rows.append(["Events (total)"] + [str(len(r["events"])) for r in ALL_RESULTS.values()])

# Event type breakdown
all_types: set[str] = set()
for r in ALL_RESULTS.values():
    for e in r["events"]:
        all_types.add(e["type"])

for etype in sorted(all_types):
    counts = []
    for r in ALL_RESULTS.values():
        n = sum(1 for e in r["events"] if e["type"] == etype)
        counts.append(str(n))
    rows.append([f"  {etype}"] + counts)

# Object class counts
for cls in sorted({k for r in ALL_RESULTS.values() for k in r.get("object_counts", {})}):
    counts = []
    for r in ALL_RESULTS.values():
        n = r.get("object_counts", {}).get(cls, 0)
        counts.append(str(n))
    rows.append([f"  obj: {cls}"] + counts)

# Print table
col_widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
print(f"\n{fmt.format(*headers)}")
print("-" * (sum(col_widths) + 3 * (len(headers) - 1)))
for row in rows:
    print(fmt.format(*row))

# Combined report
report = {
    "camera_count": len(TEST_VIDEOS),
    "camera_names": list(TEST_VIDEOS.keys()),
    "per_camera": {k: {
        "tracks": v["total_objects_tracked"],
        "detections": v["total_detections"],
        "events": len(v["events"]),
        "object_counts": v.get("object_counts", {}),
        "vehicles": v.get("vehicles", {}),
        "scene_events": v.get("scene_events", {}),
        "gate_counts": v.get("gate_counts", {}),
    } for k, v in ALL_RESULTS.items()},
    "events_by_type": {}
}

for etype in sorted(all_types):
    report["events_by_type"][etype] = {
        k: sum(1 for e in v["events"] if e["type"] == etype)
        for k, v in ALL_RESULTS.items()
    }

report_path = OUTPUT_BASE / "4feeds_report.json"
report_path.write_text(json.dumps(report, indent=2))
print(f"\nReport saved: {report_path}")
