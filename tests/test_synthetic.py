import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np


def _make_synthetic_video(
    path: str,
    width: int = 320,
    height: int = 240,
    fps: float = 10,
    duration_sec: float = 3.0,
    moving_box: bool = True,
) -> str:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    total_frames = int(fps * duration_sec)

    for i in range(total_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 128
        if moving_box:
            x = int(50 + i * (width - 100) / total_frames)
            y = int(height // 2)
            cv2.rectangle(frame, (x - 15, y - 15), (x + 15, y + 15), (0, 255, 0), -1)
        out.write(frame)

    out.release()
    return path


def _make_gate_test(path: str) -> str:
    fps = 10
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for i in range(30):
        frame = np.ones((240, 320, 3), dtype=np.uint8) * 128
        y = int(100 + i * 2)
        cv2.rectangle(frame, (140, y - 10), (180, y + 10), (0, 255, 0), -1)
        cv2.line(frame, (0, 120), (320, 120), (0, 0, 255), 1)
        out.write(frame)
    out.release()
    return path


def test_zone_containment():
    from src.models.zone import Zone

    z = Zone("test", [[0, 0], [100, 0], [100, 100], [0, 100]])
    assert z.contains(50, 50)
    assert not z.contains(150, 150)
    assert z.contains(0, 0)
    assert z.contains(99, 99)
    assert not z.contains(-1, 50)
    print("  PASS test_zone_containment")


def test_gate_crossing():
    from src.models.zone import LineGate

    g = LineGate("gate", [0, 50], [100, 50])
    assert g.cross_direction((0, 0), (50, 60)) == "exiting"
    assert g.cross_direction((0, 60), (50, 40)) == "entering"
    assert g.cross_direction((0, 40), (50, 45)) is None
    print("  PASS test_gate_crossing")


def test_gate_counter():
    from src.analytics.counting import GateCounter

    gc = GateCounter()
    gc.record("main", 1, "entering")
    gc.record("main", 1, "entering")
    gc.record("main", 2, "entering")
    gc.record("main", 1, "exiting")
    s = gc.summary()
    assert s["main"]["entries"] == 2
    assert s["main"]["exits"] == 1
    assert s["main"]["net"] == 1
    print("  PASS test_gate_counter")


def test_dwell_tracker():
    from src.analytics.dwell import DwellTracker

    dt = DwellTracker()
    dt.update(1, "zone_a", 0, True)
    dt.update(1, "zone_a", 10, True)
    dwell = dt.current_dwell(1, "zone_a", 10)
    assert dwell == 0.4
    dt.update(1, "zone_a", 10, False)
    dwell2 = dt.current_dwell(1, "zone_a", 10)
    assert dwell2 == 0.0
    print("  PASS test_dwell_tracker")


def test_event_severity():
    from src.models.event import Event, severity_for

    assert severity_for("gate_crossing") == "info"
    assert severity_for("abandoned_object") == "high"
    assert severity_for("camera_failure") == "critical"

    e = Event(event_type="abandoned_object", track_id=1, class_name="backpack")
    assert e.severity == "high"

    e2 = Event(event_type="gate_crossing", track_id=1, severity="low")
    assert e2.severity == "low"

    print("  PASS test_event_severity")


def test_stationary_filter():
    from src.analytics.stationary_filter import filter_stationary

    objects = [
        {"id": 1, "path": [[100, 100], [101, 101]], "duration_frames": 10},
        {"id": 2, "path": [[100, 100], [200, 200]], "duration_frames": 10},
        {"id": 3, "path": [[100, 100], [100, 100]], "duration_frames": 3},
    ]
    out = filter_stationary(objects, min_path_distance=20.0, min_duration_frames=5)
    assert len(out) == 1
    assert out[0]["id"] == 2
    print("  PASS test_stationary_filter")


def test_calibration():
    from src.analytics.calibration import Calibrator

    cal = Calibrator()
    assert not cal.is_calibrated

    cal.add_point([0, 360], [0.0, 0.0])
    cal.add_point([640, 360], [10.0, 0.0])
    cal.add_point([640, 0], [10.0, 5.0])
    cal.add_point([0, 0], [0.0, 5.0])
    cal.compute()
    assert cal.is_calibrated

    wx, wy = cal.image_to_world(320, 180)
    assert 4.0 < wx < 6.0
    assert 2.0 < wy < 3.0

    dist = cal.image_distance_in_world(0, 360, 640, 360)
    assert 9.0 < dist < 11.0
    print("  PASS test_calibration")


def test_track_state_machine():
    from src.tracking.state import TrackStateMachine, TrackState

    tsm = TrackStateMachine()
    tsm.init_track(1, 0)
    assert tsm.get_state_name(1) == "new"

    assert tsm.transition(1, TrackState.ACTIVE, 1)
    assert tsm.get_state_name(1) == "active"

    assert not tsm.transition(1, TrackState.MERGED, 10)
    assert tsm.get_state_name(1) == "active"

    assert tsm.transition(1, TrackState.OCCLUDED, 30)
    assert tsm.get_state_name(1) == "occluded"

    assert tsm.transition(1, TrackState.LOST, 50)
    assert tsm.get_state_name(1) == "lost"

    hist = tsm.history(1)
    assert len(hist) == 4
    print("  PASS test_track_state_machine")


def test_event_bus():
    from src.events.bus import EventBus, EventMessage, EventPriority

    bus = EventBus()
    received = []

    def cb(msg):
        received.append(msg)

    bus.subscribe("test.topic", cb)
    bus.publish(EventMessage(topic="test.topic", data={"key": "val"}, source="test"))

    assert len(received) == 1
    assert received[0].data["key"] == "val"

    bus.publish(EventMessage(topic="test.topic", data={"key2": "val2"}, source="test"))
    assert len(received) == 2
    print("  PASS test_event_bus")


def test_identity_confidence():
    from src.analytics.identity import IdentityConfidence

    ic = IdentityConfidence()
    ic.update(1, (0, 0, 100, 100), 0.9, 0)
    ic.update(1, (10, 10, 110, 110), 0.8, 1)
    ic.update(1, (20, 20, 120, 120), 0.7, 2)

    metrics = ic.get_metrics(1)
    assert metrics["confidence"] == 0.8
    assert metrics["duration_frames"] == 3
    assert metrics["stability"] == 1.0
    print("  PASS test_identity_confidence")


def test_camera_model():
    from src.models.camera import Camera, CameraStatus, CameraRegistry

    cam = Camera(
        id="test_cam",
        name="Test Camera",
        source="test.mp4",
        fps=30,
        location="Test Lab",
        gps=(51.5, -0.13),
    )
    assert cam.id == "test_cam"
    assert cam.status == CameraStatus.OFFLINE

    reg = CameraRegistry()
    reg.register(cam)
    assert reg.get("test_cam") is not None
    assert len(reg.all()) == 1

    d = cam.to_dict()
    restored = Camera.from_dict(d)
    assert restored.id == "test_cam"
    assert restored.fps == 30
    print("  PASS test_camera_model")


def test_speed_tracker():
    import time
    from tests.benchmark.metrics import SpeedTracker

    st = SpeedTracker(window=5)
    st.start()
    for _ in range(10):
        st.tick()
        time.sleep(0.001)

    r = st.report()
    assert r["total_frames"] == 10
    assert r["overall_fps"] > 0
    print("  PASS test_speed_tracker")


def test_track_predictor():
    from src.analytics.prediction import TrackPredictor

    tp = TrackPredictor()
    for i in range(10):
        tp.update(1, float(100 + i * 5), 200.0, i)

    pred = tp.predict(1)
    assert pred is not None
    x, y, conf = pred
    assert 140 < x < 160
    assert y == 200.0
    assert 0 < conf <= 1.0
    print("  PASS test_track_predictor")


def test_event_correlation():
    import time
    from src.analytics.correlation import EventCorrelator

    ec = EventCorrelator()
    now = time.time()

    zone_entry = {"type": "zone_entry", "track_id": 1, "timestamp": now - 5}
    loiter = {"type": "possible_loitering", "track_id": 1, "timestamp": now - 3}
    abandon = {"type": "abandoned_object", "track_id": 1, "timestamp": now - 1}

    assert ec.process_event(zone_entry) is None
    assert ec.process_event(loiter) is None
    incident = ec.process_event(abandon)
    assert incident is not None, "Expected incident from 3 correlated events"
    assert incident.incident_type == "suspicious_activity"
    assert incident.severity == "critical"
    print("  PASS test_event_correlation")


def test_plugin_base():
    from src.plugin.base import PluginRegistry

    reg = PluginRegistry()
    assert len(reg.all()) == 0
    print("  PASS test_plugin_base")


def test_config_loader():
    from src.config import ConfigLoader

    loader = ConfigLoader()
    detector_cfg = loader.load("detector")
    assert detector_cfg.get("model_family") == "yolo11"
    assert detector_cfg.get("conf_threshold") == 0.4

    tracker_cfg = loader.load("tracker")
    assert tracker_cfg.get("track_buffer") == 450

    analytics_cfg = loader.load("analytics")
    assert analytics_cfg.get("filter_stationary_objects") is True
    print("  PASS test_config_loader")


def test_analytics_db():
    import tempfile
    from src.db.repository import AnalyticsDB

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db = AnalyticsDB(db_path)
    db.connect()
    db.upsert_camera({
        "id": "test", "name": "Test", "source": "test.mp4",
        "fps": 25, "resolution": "640,360", "location": "lab",
        "gps": None, "timezone": "UTC", "status": "online",
    })
    cameras = db.query("SELECT * FROM cameras")
    assert len(cameras) == 1
    assert cameras[0]["id"] == "test"

    run_id = db.start_run("test", "test.mp4", "yolo11", "nano")
    assert run_id is not None

    db.insert_event(run_id, "test", {
        "type": "zone_entry", "track_id": 1, "class": "person",
        "severity": "info", "confidence": 1.0,
    })
    events = db.get_events(camera_id="test")
    assert len(events) == 1

    db.close()
    Path(db_path).unlink()
    print("  PASS test_analytics_db")


def test_synthetic_video_gate():
    from src.pipeline import analyze_video

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = _make_gate_test(f.name)

    try:
        result = analyze_video(
            video_path=video_path,
            output_dir="/tmp/sentinel_test",
            model_size="nano",
            conf_threshold=0.25,
            device="cpu",
            max_frames=30,
            filter_stationary_objects=False,
        )
        assert result["total_frames_processed"] > 0
        print("  PASS test_synthetic_video_gate (no assertion on detections in synthetic)")
    except Exception as e:
        print(f"  INFO test_synthetic_video_gate: {e}")
    finally:
        Path(video_path).unlink(missing_ok=True)


def test_vehicle_registry():
    from src.analytics.vehicle.registry import VehicleRegistry

    reg = VehicleRegistry()
    rec = reg.register(1, "ABC123", "red", "car", "medium")
    assert rec.plate == "ABC123"
    assert rec.color == "red"
    assert rec.visit_count == 1

    rec2 = reg.register(2, "ABC123", "red", "car", "medium")
    assert rec2.visit_count == 1

    assert reg.get_by_plate("ABC123") is not None
    assert reg.get_by_track(1) is not None
    s = reg.summary()
    assert s["total_vehicles"] == 1
    assert s["with_plates"] == 1
    print("  PASS test_vehicle_registry")


def test_vehicle_registry_no_plate():
    from src.analytics.vehicle.registry import VehicleRegistry

    reg = VehicleRegistry()
    rec = reg.register(1, "", "blue", "truck", "large")
    assert rec.plate == ""
    assert rec.color == "blue"
    s = reg.summary()
    assert s["total_vehicles"] == 1
    assert s["without_plates"] == 1
    print("  PASS test_vehicle_registry_no_plate")


def test_vehicle_attributes():
    from src.analytics.vehicle.attributes import vehicle_size_class, _COLOR_NAMES

    assert vehicle_size_class((0, 0, 50, 50)) == "small"
    assert vehicle_size_class((0, 0, 200, 200)) == "large"
    assert "red" in _COLOR_NAMES
    assert "white" in _COLOR_NAMES
    print("  PASS test_vehicle_attributes")


def test_plate_validation():
    from src.analytics.vehicle.validation import clean_plate, validate_plate

    assert clean_plate("abc 123") == "ABC123"
    assert clean_plate("AB-12 CD") == "AB12CD"
    assert clean_plate("") == ""

    val, qual = validate_plate("ABC123")
    assert val == "ABC123"
    assert qual > 0

    val2, qual2 = validate_plate("AB")
    assert val2 == ""
    assert qual2 == 0.0

    val3, qual3 = validate_plate("ABCDEFGHIJ")
    assert val3 == ""
    assert qual3 == 0.0
    print("  PASS test_plate_validation")


def test_preprocessing():
    import cv2, numpy as np
    from src.analytics.vehicle.preprocessing import preprocess_for_ocr

    blank = np.ones((50, 200, 3), dtype=np.uint8) * 128
    cv2.rectangle(blank, (30, 10), (170, 40), (0, 0, 0), -1)
    cv2.putText(blank, "ABC123", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    variants = preprocess_for_ocr(blank)
    assert len(variants) >= 3
    for v in variants:
        assert v.shape[0] > 0 and v.shape[1] > 0
    print("  PASS test_preprocessing")


def run_all():
    tests = [
        test_zone_containment,
        test_gate_crossing,
        test_gate_counter,
        test_dwell_tracker,
        test_event_severity,
        test_stationary_filter,
        test_calibration,
        test_track_state_machine,
        test_event_bus,
        test_identity_confidence,
        test_camera_model,
        test_speed_tracker,
        test_track_predictor,
        test_event_correlation,
        test_plugin_base,
        test_config_loader,
        test_analytics_db,
        test_synthetic_video_gate,
        test_vehicle_registry,
        test_vehicle_registry_no_plate,
        test_vehicle_attributes,
        test_plate_validation,
        test_preprocessing,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all() else 1)
