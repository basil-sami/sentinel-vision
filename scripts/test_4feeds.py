"""Multi-feed test: process 4 cameras concurrently via MultiCameraPipeline."""

import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from src.optimization.multi_stream import MultiCameraPipeline
from src.visualization.mosaic import create_mosaic

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

zone_config = json.loads(ZONE_CONFIG.read_text())
calib_config = {}
if CALIB_CONFIG.exists():
    calib_config = json.loads(CALIB_CONFIG.read_text())

camera_configs = []
for name, rel_path in TEST_VIDEOS.items():
    camera_configs.append({
        "video_path": str(SRC_DIR / rel_path),
        "name": name,
        "model_family": "yolo11",
        "model_size": "nano",
        "conf_threshold": 0.4,
        "device": "cpu",
        "max_frames": 200,
        "zone_config": zone_config,
        "calibration_config": calib_config,
        "capture_evidence": True,
        "filter_stationary_objects": True,
        "min_move_distance": 20.0,
    })

pipeline = MultiCameraPipeline(
    camera_configs=camera_configs,
    output_dir=str(OUTPUT_BASE),
)

print("Starting 4 feeds concurrently...")
report = pipeline.run()
print(f"\nAll done! Total events across cameras: {report['event_count']}")

# --- Generate side-by-side mosaic ---
print("\nGenerating mosaic...")
mosaic_inputs = {}
for cam_key, cam_result in report.get("per_camera", {}).items():
    out_vid = cam_result.get("output_video", "")
    if out_vid and Path(out_vid).exists():
        cam_name = cam_key
        mosaic_inputs[cam_name] = out_vid
        print(f"  Added: {cam_name} ({Path(out_vid).stat().st_size/1e6:.1f}MB)")

if len(mosaic_inputs) == len(TEST_VIDEOS):
    create_mosaic(
        mosaic_inputs,
        output_path=str(OUTPUT_BASE / "mosaic_4feeds.mp4"),
        layout="2x2",
    )
else:
    print(f"  Skipping mosaic — only {len(mosaic_inputs)}/{len(TEST_VIDEOS)} outputs")
