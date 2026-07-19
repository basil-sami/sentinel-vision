from pathlib import Path
from tqdm import tqdm

from src.video import VideoLoader
from src.detection import YOLODetector
from src.tracking.tracker import Tracker
from src.analytics.object_history import ObjectHistory
from src.visualization import Annotator


def analyze_video(
    video_path: str,
    output_dir: str = "outputs",
    model_size: str = "nano",
    conf_threshold: float = 0.5,
    device: str = "cpu",
    max_frames: int | None = None,
    track_thresh: float = 0.5,
    match_thresh: float = 0.8,
    track_buffer: int = 30,
    trail_length: int = 50,
    use_reid: bool = True,
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

    output_video_path = str(output_dir / "output_tracking.mp4")
    annotator = Annotator(
        output_path=output_video_path,
        fps=loader.fps,
        width=loader.width,
        height=loader.height,
    )

    total_frames = min(loader.frame_count, max_frames) if max_frames else loader.frame_count
    pbar = tqdm(total=total_frames, desc="Processing video")

    for i, frame in enumerate(loader):
        if max_frames and i >= max_frames:
            break

        detections = detector.detect(frame, conf_threshold=conf_threshold)
        tracks = tracker.update(detections, frame)
        history.update(tracks, i)

        annotated = annotator.draw_tracks(frame, tracks, history, trail_length=trail_length)
        annotator.write_frame(annotated)
        pbar.update(1)

    pbar.close()
    annotator.release()
    loader.release()

    objects_export = history.export()
    total_tracked = len(objects_export)
    all_detection_count = sum(len(o["path"]) for o in objects_export)

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
