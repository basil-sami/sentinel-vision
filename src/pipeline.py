import logging
from pathlib import Path
import cv2
from tqdm import tqdm

from src.video import VideoLoader
from src.detection import YOLODetector
from src.tracking.tracker import Tracker
from src.analytics.object_history import ObjectHistory
from src.analytics.merge_fragments import merge_fragments
from src.analytics.stationary_filter import filter_stationary
from src.analytics.zones import ZoneManager
from src.analytics.counting import GateCounter
from src.analytics.dwell import DwellTracker
from src.analytics.events import EventDetector
from src.analytics.abandoned import AbandonedDetector
from src.analytics.movement import movement_stats
from src.analytics.calibration import Calibrator
from src.analytics.interaction import InteractionModel
from src.analytics.evidence import EvidenceCapture
from src.analytics.vehicle.orchestrator import VehicleAnalyzer
from src.analytics.scene.orchestrator import SceneAnalyzer
from src.analytics.identity import IdentityConfidence
from src.analytics.prediction import TrackPredictor
from src.analytics.correlation import EventCorrelator
from src.analytics.time_sync import TimeSync
from src.analytics.face_recognition import FaceRecognizer
from src.models.event import EventStore, Event
from src.visualization import Annotator
from src.visualization.zone_renderer import draw_zones, draw_gates, draw_event_ticker


def analyze_video(
    video_path: str,
    output_dir: str = "outputs",
    model_family: str = "yolo11",
    model_size: str = "nano",
    conf_threshold: float = 0.4,
    device: str = "cpu",
    max_frames: int | None = None,
    track_thresh: float = 0.4,
    match_thresh: float = 0.7,
    track_low_thresh: float = 0.1,
    track_buffer: int = 450,
    trail_length: int = 50,
    use_reid: bool = True,
    reid_model: str = "x1_0",
    zone_config: dict | None = None,
    calibration_config: dict | None = None,
    capture_evidence: bool = True,
    filter_stationary_objects: bool = True,
    min_move_distance: float = 20.0,
    target_classes: dict[int, str] | None = None,
    use_tensorrt: bool = False,
    log_level: int = logging.WARNING,
    plate_read_interval: int = 10,
    use_cmc: bool = False,
    reid_refresh_interval: int = 50,
    reid_new_track_frames: int = 3,
    detector: "YOLODetector | None" = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger(f"pipeline.{Path(video_path).stem}")
    log.setLevel(log_level)
    log.info("=== Pipeline start ===")
    log.info("video=%s  model=%s/%s  device=%s  tensorrt=%s  max_frames=%s",
             video_path, model_family, model_size, device, use_tensorrt, max_frames)

    loader = VideoLoader(video_path)
    log.info("Video: %d frames, %.2f fps, %dx%d",
             loader.frame_count, loader.fps, loader.width, loader.height)
    if detector is None:
        detector = YOLODetector(
            model_family=model_family,
            model_size=model_size,
            device=device,
            target_classes=target_classes,
            use_tensorrt=use_tensorrt,
        )
    tracker = Tracker(
        track_thresh=track_thresh,
        track_low_thresh=track_low_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        use_reid=use_reid,
        reid_model=reid_model,
        device=device,
        use_cmc=use_cmc,
        reid_refresh_interval=reid_refresh_interval,
        reid_new_track_frames=reid_new_track_frames,
    )
    history = ObjectHistory()
    events = EventStore()

    zone_mgr = ZoneManager()
    if zone_config:
        zone_mgr = ZoneManager.from_config(zone_config)

    calibrator = Calibrator()
    if calibration_config:
        calibrator = Calibrator.from_config(calibration_config)

    gate_counter = GateCounter()
    dwell_tracker = DwellTracker()
    event_detector = EventDetector()
    abandoned_detector = AbandonedDetector(stationary_threshold_frames=track_buffer)
    interaction_model = InteractionModel()
    vehicle_analyzer = VehicleAnalyzer(plate_read_interval=plate_read_interval)
    scene_analyzer = SceneAnalyzer()
    identity_tracker = IdentityConfidence()
    predictor = TrackPredictor()
    correlator = EventCorrelator()
    time_sync = TimeSync(fps=loader.fps)
    face_recognizer = FaceRecognizer(device=device)

    output_video_path = str(output_dir / "output_tracking.mp4")
    annotator = Annotator(
        output_path=output_video_path,
        fps=loader.fps,
        width=loader.width,
        height=loader.height,
    )

    evidence = None
    if capture_evidence:
        evidence = EvidenceCapture(
            str(output_dir),
            fps=loader.fps,
            width=loader.width,
            height=loader.height,
        )

    total_frames = min(loader.frame_count, max_frames) if max_frames else loader.frame_count
    pbar = tqdm(total=total_frames, desc="Processing video")

    _zone_state: dict[int, set[str]] = {}

    for i, frame in enumerate(loader):
        if max_frames and i >= max_frames:
            log.info("Reached max_frames=%d, stopping", max_frames)
            break

        detections = detector.detect(frame, conf_threshold=conf_threshold)
        tracks = tracker.update(detections, frame, frame_index=i)
        history.update(tracks, i)

        if i == 0:
            log.info("Frame 0: %d detections, %d tracks", len(detections), len(tracks))
        if len(detections) > 0 and i % 50 == 0:
            classes = [d.class_name for d in detections[:5]]
            log.debug("Frame %d: %d detections [%s], %d tracks",
                      i, len(detections), ",".join(classes[:3]), len(tracks))

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if evidence:
            evidence.add_frame(frame_bgr)

        for t in tracks:
            cx = (t.bbox[0] + t.bbox[2]) // 2
            cy = (t.bbox[1] + t.bbox[3]) // 2

            current_zones = set(z.name for z in zone_mgr.zones_at(cx, cy))
            prev_zones = _zone_state.get(t.id, set())

            for zn in current_zones - prev_zones:
                ev = event_detector.check_zone_entry(t.id, t.class_name, zn, cx, cy)
                events.add(ev)
                if evidence:
                    evidence.capture_for_event("zone_entry", t.id, {"zone": zn, "class": t.class_name})
            for zn in prev_zones - current_zones:
                dwell_s = dwell_tracker.current_dwell(t.id, zn, i, loader.fps)
                ev = event_detector.check_zone_exit(t.id, t.class_name, zn, dwell_s, cx, cy)
                events.add(ev)

            for zn in current_zones:
                dwell_tracker.update(t.id, zn, i, True)
                dwell_s = dwell_tracker.current_dwell(t.id, zn, i, loader.fps)
                loiter_ev = event_detector.check_loitering(t.id, t.class_name, zn, dwell_s, i, cx, cy)
                if loiter_ev:
                    loiter_ev.confidence = min(dwell_s / 1200.0, 1.0)
                    events.add(loiter_ev)
                    if evidence:
                        evidence.capture_for_event("loitering", t.id, {"zone": zn, "duration": dwell_s})
            for zn in prev_zones - current_zones:
                dwell_tracker.update(t.id, zn, i, False)

            _zone_state[t.id] = current_zones

            for gate_name, direction in zone_mgr.check_gate_crossing(t.id, cx, cy):
                gate_counter.record(gate_name, t.id, direction)
                ev = Event(
                    event_type="gate_crossing",
                    track_id=t.id,
                    class_name=t.class_name,
                    zone=gate_name,
                    location=[cx, cy],
                    message=f"{t.class_name} ID {t.id} {'entered' if direction == 'entering' else 'exited'} via {gate_name}",
                )
                events.add(ev)

            if t.class_name != "person":
                ab_ev = abandoned_detector.update(t.id, t.class_name, t.bbox, i, tracks)
                if ab_ev:
                    ab_ev.severity = "high"
                    events.add(ab_ev)
                    if evidence:
                        evidence.capture_for_event("abandoned_object", t.id, {"class": t.class_name, "duration": ab_ev.duration})

        # Interaction model
        interaction_events = interaction_model.update(tracks, i)
        for iev in interaction_events:
            events.add(iev)

        # Vehicle intelligence
        vehicle_events = vehicle_analyzer.process_frame(frame, tracks, i, calibrator)
        for ve in vehicle_events:
            events.add(ve)

        # Scene understanding (carrying, overloaded vehicles)
        scene_events = scene_analyzer.process_frame(tracks, i, calibrator, zone_mgr)
        for se in scene_events:
            events.add(se)

        # Identity confidence tracking per track
        for t in tracks:
            identity_tracker.update(t.id, t.bbox, t.confidence, i)
            cx = (t.bbox[0] + t.bbox[2]) // 2
            cy = (t.bbox[1] + t.bbox[3]) // 2
            predictor.update(t.id, cx, cy, i)

        # Time sync
        ts = time_sync.frame_timestamp(i)

        # Face recognition (person tracks only)
        face_events = face_recognizer.process_frame(frame, tracks, i)
        for fe in face_events:
            events.add(Event(
                event_type=fe["type"],
                track_id=fe["track_id"],
                class_name="person",
                message=f"Recognized {fe['name']} (track {fe['track_id']}, confidence={fe['confidence']})",
            ))

        # Event correlation
        for ev in events.recent(5):
            inc = correlator.process_event({
                "type": ev.event_type,
                "track_id": ev.track_id,
                "timestamp": ts.utc_timestamp,
                "severity": ev.severity,
            })
            if inc:
                events.add(Event(
                    event_type=f"incident_{inc.incident_type}",
                    track_id=list(inc.track_ids)[0] if inc.track_ids else -1,
                    class_name="system",
                    severity=inc.severity,
                    message=inc.summary,
                ))

        # Render
        face_ids = face_recognizer.get_all_identities()
        annotated = annotator.draw_tracks(
            frame, tracks, history, trail_length=trail_length,
            identities=face_ids if face_ids else None,
        )
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        active_zone_names = set()
        for zset in _zone_state.values():
            active_zone_names.update(zset)

        if zone_mgr.zones:
            annotated_bgr = draw_zones(annotated_bgr, zone_mgr.zones, active_zone_names, gate_counter.summary())
        if zone_mgr.gates:
            annotated_bgr = draw_gates(annotated_bgr, zone_mgr.gates)

        annotated_bgr = draw_event_ticker(annotated_bgr, events.all())

        annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        annotator.write_frame(annotated)
        pbar.update(1)

    pbar.close()
    annotator.release()
    loader.release()

    log.info("Frames processed: %d", i + 1)
    log.info("Raw tracks: %d", len(history.export()))

    objects_export = history.export()

    if use_reid and objects_export:
        log.info("Merging fragments...")
        objects_export = merge_fragments(
            objects_export, track_buffer=track_buffer
        )
        log.info("After merge: %d tracks", len(objects_export))

    if filter_stationary_objects and objects_export:
        log.info("Filtering stationary (min_move=%.1f)...", min_move_distance)
        objects_export = filter_stationary(
            objects_export,
            min_path_distance=min_move_distance,
            min_duration_frames=5,
        )

    total_tracked = len(objects_export)
    all_detection_count = sum(len(o["path"]) for o in objects_export)

    for obj in objects_export:
        stats = movement_stats(obj.get("path", []))
        if calibrator.is_calibrated:
            path = obj.get("path", [])
            stats["distance_meters"] = calibrator.path_length_in_world(path)
            stats["speed_mps"] = calibrator.speed_in_world(path, loader.fps)
        obj["movement"] = stats

    result = {
        "video": str(Path(video_path).name),
        "video_duration_sec": round(loader.duration, 2),
        "total_frames_processed": total_frames,
        "fps": loader.fps,
        "resolution": f"{loader.width}x{loader.height}",
        "total_objects_tracked": total_tracked,
        "total_detections": all_detection_count,
        "object_counts": history.summary()["by_class"],
        "objects": objects_export,
        "output_video": output_video_path,
        "zones": zone_mgr.get_config(),
        "calibration": calibrator.get_config(),
        "gate_counts": gate_counter.summary(),
        "dwell_summary": dwell_tracker.summary(),
        "events": events.export(),
        "vehicles": vehicle_analyzer.get_registry().summary(),
        "vehicle_list": [v.to_dict() for v in vehicle_analyzer.get_registry().all()],
        "scene_events": {
            "person_carrying": len(events.by_type("person_carrying")),
            "overloaded_vehicle": len(events.by_type("overloaded_vehicle")),
        },
        "identities": [
            {"track_id": tid, "name": name, "confidence": conf}
            for tid, (name, conf) in face_recognizer.get_all_identities().items()
        ],
        "incidents": [inc.to_dict() for inc in correlator.incidents()],
    }

    if evidence:
        result["evidence_clips"] = evidence.list_captures()

    import json
    analytics_path = output_dir / "analytics.json"
    analytics_path.write_text(json.dumps(result, indent=2))

    summary_path = output_dir / "summary.txt"
    summary_lines = [
        f"Sentinel Vision — Video Analysis Summary",
        f"{'=' * 40}",
        f"Video: {result['video']}",
        f"Duration: {result['video_duration_sec']} seconds",
        f"Resolution: {result['resolution']}",
        f"Frames processed: {result['total_frames_processed']}",
        f"",
        f"Unique objects tracked: {total_tracked}",
        f"Total detections (across frames): {all_detection_count}",
        f"",
        f"Objects by class:",
    ]
    for cls, count in sorted(history.summary()["by_class"].items()):
        summary_lines.append(f"  {cls}: {count}")
    summary_lines.append(f"")

    if calibrator.is_calibrated:
        summary_lines.append(f"Calibration: Active (world coordinates enabled)")
        summary_lines.append(f"")

    if zone_mgr.zones:
        summary_lines.append(f"Zone Config:")
        for z in zone_mgr.zones:
            summary_lines.append(f"  {z.name} ({z.zone_type})")
        summary_lines.append(f"")

    if gate_counter.summary():
        summary_lines.append(f"Gate Counts:")
        for gname, counts in gate_counter.summary().items():
            summary_lines.append(f"  {gname}: +{counts['entries']}  -{counts['exits']}  (net {counts['net']:+d})")
        summary_lines.append(f"")

    critical_events = events.critical()
    high_events = events.by_severity("high")
    if critical_events or high_events:
        summary_lines.append(f"High-Severity Events:")
        for ev in (critical_events + high_events):
            summary_lines.append(f"  [{ev.severity.upper()}] {ev.message}")
        summary_lines.append(f"")

    scene_events_list = [e for e in events.all() if e.event_type in ("person_carrying", "overloaded_vehicle")]
    if scene_events_list:
        summary_lines.append(f"Scene Events ({len(scene_events_list)}):")
        for ev in scene_events_list:
            summary_lines.append(f"  [{ev.event_type}] {ev.message}")
        summary_lines.append(f"")

    if events.all():
        summary_lines.append(f"Events ({len(events.all())} total):")
        for ev in events.all()[-15:]:
            summary_lines.append(f"  [{ev.event_type}] {ev.message}")
        summary_lines.append(f"")

    # Vehicle intelligence
    v = result.get("vehicles", {})
    if v.get("total_vehicles", 0) > 0:
        summary_lines.append(f"Vehicle Intelligence:")
        summary_lines.append(f"  Total:      {v['total_vehicles']}")
        summary_lines.append(f"  With plates: {v['with_plates']}")
        summary_lines.append(f"  Without:    {v['without_plates']}")
        summary_lines.append(f"  Visits:     {v['total_visits']}")
        for vl in result.get("vehicle_list", [])[:5]:
            plate = vl.get("plate") or "no-plate"
            color = vl.get("color", "unknown")
            vtype = vl.get("vehicle_type", "unknown")
            summary_lines.append(f"  {plate:>10s}  {color:>8s}  {vtype:>10s}  {vl['visit_count']} visit(s)")
        summary_lines.append(f"")

    # Face recognitions
    idents = result.get("identities", [])
    if idents:
        summary_lines.append(f"Face Recognitions ({len(idents)}):")
        for id_ in idents:
            summary_lines.append(f"  Track {id_['track_id']}: {id_['name']} (conf={id_['confidence']})")
        summary_lines.append(f"")

    # Correlated incidents
    incs = result.get("incidents", [])
    if incs:
        summary_lines.append(f"Correlated Incidents ({len(incs)}):")
        for inc in incs:
            summary_lines.append(f"  [{inc['severity'].upper()}] {inc['incident_type']}: {inc['summary']}")
        summary_lines.append(f"")

    if objects_export:
        longest = max(objects_export, key=lambda o: o["duration_frames"])
        summary_lines.append(
            f"Longest tracked object: ID {longest['id']} "
            f"({longest['class']}) — {longest['duration_frames']} frames"
        )
    summary_lines.append(f"")
    summary_lines.append(f"Annotated video: {output_video_path}")
    summary_lines.append(f"JSON report: {analytics_path}")
    summary_path.write_text("\n".join(summary_lines))

    log.info("=== Pipeline complete ===")
    log.info("Final: %d tracks, %d detections, %d events, %s",
             total_tracked, all_detection_count, len(events.all()), output_video_path)
    print(f"\nSummary written to {summary_path}")
    return result
