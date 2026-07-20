from pathlib import Path
import cv2
import numpy as np


def create_mosaic(
    video_paths: dict[str, str],
    output_path: str,
    layout: str = "2x2",
    max_frames: int | None = None,
    label_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (30, 30, 30),
) -> str:
    """Stitch N videos into a side-by-side composite.

    Args:
        video_paths: {label: path_to_video}
        output_path: Where to save the mosaic.
        layout: "2x2" or "horizontal" or "vertical".
        max_frames: Max frames to render (None = shortest video).
        label_color: BGR color for labels.
        bg_color: BGR background between tiles.

    Returns:
        Path to output mosaic video.
    """
    labels = list(video_paths.keys())
    paths = [video_paths[k] for k in labels]

    caps = [cv2.VideoCapture(p) for p in paths]
    widths = []
    heights = []
    fpses = []
    for cap in caps:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        widths.append(w)
        heights.append(h)
        fpses.append(fps)

    target_fps = max(fpses) if fpses else 25
    n = len(caps)

    if layout == "2x2" and n == 4:
        cols, rows = 2, 2
    elif layout == "horizontal":
        cols, rows = n, 1
    elif layout == "vertical":
        cols, rows = 1, n
    else:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

    tile_w = max(widths)
    tile_h = max(heights)
    gap = 2
    canvas_w = cols * tile_w + (cols + 1) * gap
    canvas_h = rows * tile_h + (rows + 1) * gap

    if max_frames is None:
        max_frames = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (canvas_w, canvas_h))

    for frame_i in range(max_frames):
        canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

        for idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

            if frame.shape[1] != tile_w or frame.shape[0] != tile_h:
                frame = cv2.resize(frame, (tile_w, tile_h))

            col = idx % cols
            row = idx // cols
            x = gap + col * (tile_w + gap)
            y = gap + row * (tile_h + gap)
            canvas[y:y + tile_h, x:x + tile_w] = frame

            label = labels[idx] if idx < len(labels) else f"cam{idx}"
            cv2.putText(
                canvas, label, (x + 8, y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2,
            )

        out.write(canvas)

        if frame_i % 50 == 0:
            print(f"  Mosaic frame {frame_i}/{max_frames}")

    for cap in caps:
        cap.release()
    out.release()
    print(f"Mosaic saved: {output_path}")
    return output_path
