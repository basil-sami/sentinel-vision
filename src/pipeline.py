from pathlib import Path
from tqdm import tqdm

from src.video import VideoLoader
from src.detection import YOLODetector
from src.visualization import Annotator


def analyze_video(
    video_path: str,
    output_dir: str = "outputs",
    model_size: str = "nano",
    conf_threshold: float = 0.5,
    device: str = "cpu",
    max_frames: int | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = VideoLoader(video_path)
    detector = YOLODetector(model_size=model_size, device=device)

    output_video_path = str(output_dir / "output_tracking.mp4")
    annotator = Annotator(
        output_path=output_video_path,
        fps=loader.fps,
        width=loader.width,
        height=loader.height,
    )

    total_frames = min(loader.frame_count, max_frames) if max_frames else loader.frame_count
    object_counts: dict[str, int] = {}
    all_detections: list[dict] = []

    pbar = tqdm(total=total_frames, desc="Processing video")
    for i, frame in enumerate(loader):
        if max_frames and i >= max_frames:
            break

        detections = detector.detect(frame, conf_threshold=conf_threshold)

        for det in detections:
            object_counts[det.class_name] = object_counts.get(det.class_name, 0) + 1
            all_detections.append({
                "frame": i,
                "time": round(i / loader.fps, 2),
                **det.to_dict(),
            })

        annotated = annotator.draw_detections(frame, detections)
        annotator.write_frame(annotated)
        pbar.update(1)

    pbar.close()
    annotator.release()
    loader.release()

    result = {
        "video": str(Path(video_path).name),
        "video_duration_sec": round(loader.duration, 2),
        "total_frames_processed": total_frames,
        "fps": loader.fps,
        "resolution": f"{loader.width}x{loader.height}",
        "total_detections": len(all_detections),
        "object_counts": object_counts,
        "detections": all_detections,
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
        f"Object detections (counted per frame):",
    ]
    for cls, count in sorted(object_counts.items()):
        summary_lines.append(f"  {cls}: {count}")
    summary_lines.append(f"")
    summary_lines.append(f"Annotated video: {output_video_path}")
    summary_lines.append(f"JSON report: {analytics_path}")
    summary_path.write_text("\n".join(summary_lines))

    print(f"\nSummary written to {summary_path}")
    return result
