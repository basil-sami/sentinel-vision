from pathlib import Path
import cv2
from tqdm import tqdm

from src.video import VideoLoader
from src.detection import YOLODetector
from src.tracking.tracker import Tracker
from src.analytics.object_history import ObjectHistory
from src.analytics.merge_fragments import merge_fragments
from src.analytics.zones import ZoneManager
from src.analytics.counting import GateCounter
from src.analytics.dwell import DwellTracker
from src.analytics.events import EventDetector
from src.analytics.abandoned import AbandonedDetector
from src.analytics.movement import movement_stats
from src.models.event import EventStore, Event
from src.visualization import Annotator
from src.visualization.zone_renderer import draw_zones, draw_gates


def analyze_video(
    video_path: str,
    output_dir: str = "outputs",
    model_size: str = "nano",
    conf_threshold: float = 0.25,
    device: str = "cpu",
    max_frames: int | None = None,
    track_thresh: float = 0.5,
    match_thresh: float = 0.8,
    track_buffer: int = 300,
    trail_length: int = 50,
    use_reid: bool = True,
    zone_config: dict | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = VideoLoader(video_path)
    detector = YOLODetector(model_size=model_size, device=device)
    tracker = Tracker(
        track_thresh=track_thresh,
        match_thresh=match_thresh,
        track_buffer=track_buffer,
        use_reid=use_reid,
        device=device,
    )
    history = ObjectHistory()
    events = EventStore()

    zone_mgr = ZoneManager()
    if zone_config:
        zone_mgr = ZoneManager.from_config(zone_config)

    gate_counter = GateCounter()
    dwell_tracker = DwellTracker()
    event_detector = EventDetector()
    abandoned_detector = AbandonedDetector(stationary_threshold_frames=track_buffer)

    output_video_path = str(output_dir / "output_tracking.mp4")
    annotator = Annotator(
        output_path=output_video_path,
        fps=loader.fps,
        width=loader.width,
        height=loader.height,
    )

    total_frames = min(loader.frame_count, max_frames) if max_frames else loader.frame_count
    pbar = tqdm(total=total_frames, desc="Processing video")

    # Track which zones each object is currently inside
    _zone_state: dict[int, set[str]] = {}

    for i, frame in enumerate(loader):
        if max_frames and i >= max_frames:
            break

        detections = detector.detect(frame, conf_threshold=conf_threshold)
        tracks = tracker.update(detections, frame)
        history.update(tracks, i)

        for t in tracks:
            cx = (t.bbox[0] + t.bbox[2]) // 2
            cy = (t.bbox[1] + t.bbox[3]) // 2

            # Zone entry/exit
            current_zones = set(z.name for z in zone_mgr.zones_at(cx, cy))
            prev_zones = _zone_state.get(t.id, set())

            for zn in current_zones - prev_zones:
                events.add(event_detector.check_zone_entry(t.id, t.class_name, zn, cx, cy))
            for zn in prev_zones - current_zones:
                dwell_s = dwell_tracker.current_dwell(t.id, zn, i, loader.fps)
                events.add(event_detector.check_zone_exit(t.id, t.class_name, zn, dwell_s, cx, cy))

            for zn in current_zones:
                dwell_tracker.update(t.id, zn, i, True)
                dwell_s = dwell_tracker.current_dwell(t.id, zn, i, loader.fps)
                loiter_ev = event_detector.check_loitering(t.id, t.class_name, zn, dwell_s, i, cx, cy)
                if loiter_ev:
                    events.add(loiter_ev)
            for zn in prev_zones - current_zones:
                dwell_tracker.update(t.id, zn, i, False)

            _zone_state[t.id] = current_zones

            # Gate crossing
            for gate_name, direction in zone_mgr.check_gate_crossing(t.id, cx, cy):
                gate_counter.record(gate_name, t.id, direction)
                if direction == "entering":
                    events.add(Event(
                        event_type="gate_crossing",
                        track_id=t.id,
                        class_name=t.class_name,
                        zone=gate_name,
                        location=[cx, cy],
                        message=f"{t.class_name} ID {t.id} entered via {gate_name}",
                    ))
                else:
                    events.add(Event(
                        event_type="gate_crossing",
                        track_id=t.id,
                        class_name=t.class_name,
                        zone=gate_name,
                        location=[cx, cy],
                        message=f"{t.class_name} ID {t.id} exited via {gate_name}",
                    ))

            # Abandoned object detection
            if t.class_name != "person":
                ab_ev = abandoned_detector.update(t.id, t.class_name, t.bbox, i, tracks)
                if ab_ev:
                    events.add(ab_ev)

        # Render frame
        annotated = annotator.draw_tracks(frame, tracks, history, trail_length=trail_length)
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        if zone_mgr.zones:
            annotated_bgr = draw_zones(annotated_bgr, zone_mgr.zones, gate_counter.summary())
        if zone_mgr.gates:
            annotated_bgr = draw_gates(annotated_bgr, zone_mgr.gates)

        recent_events = events.all()[-5:]
        y_offset = annotated_bgr.shape[0] - 20 * len(recent_events) - 10
        for ev in recent_events:
            msg = ev.message[:60]
            cv2.putText(annotated_bgr, msg, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
            y_offset += 18

        annotated = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        annotator.write_frame(annotated)
        pbar.update(1)

    pbar.close()
    annotator.release()
    loader.release()

    objects_export = history.export()

    if use_reid and objects_export:
        objects_export = merge_fragments(
            objects_export, track_buffer=track_buffer
        )

    total_tracked = len(objects_export)
    all_detection_count = sum(len(o["path"]) for o in objects_export)

    for obj in objects_export:
        obj["movement"] = movement_stats(obj.get("path", []))

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
        "gate_counts": gate_counter.summary(),
        "dwell_summary": dwell_tracker.summary(),
        "events": events.export(),
    }

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

    if events.all():
        summary_lines.append(f"Events ({len(events.all())} total):")
        for ev in events.all()[-10:]:
            summary_lines.append(f"  [{ev.event_type}] {ev.message}")
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

    print(f"\nSummary written to {summary_path}")
    return result
