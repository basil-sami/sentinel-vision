import cv2
import numpy as np


def compute_sharpness(crop: np.ndarray) -> float:
    if crop.size == 0 or min(crop.shape[:2]) < 5:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_plate_size(plate_bbox: tuple[int, int, int, int], frame_area: int) -> float:
    x1, y1, x2, y2 = plate_bbox
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    area = w * h
    return area / max(frame_area, 1)


def compute_viewing_angle(plate_bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = plate_bbox
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    if h == 0:
        return 0.0
    aspect = w / h
    ideal = 4.0
    deviation = abs(aspect - ideal) / ideal
    return max(0.0, 1.0 - deviation)


def compute_brightness_contrast(crop: np.ndarray) -> tuple[float, float]:
    if crop.size == 0:
        return (0.0, 0.0)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    brightness = 1.0 - abs(mean - 127.0) / 127.0
    contrast = min(std / 64.0, 1.0)
    return (brightness, contrast)


def estimate_motion_blur(prev_crop: np.ndarray, curr_crop: np.ndarray) -> float:
    if prev_crop.size == 0 or curr_crop.size == 0:
        return 1.0
    h, w = min(prev_crop.shape[0], curr_crop.shape[0]), min(prev_crop.shape[1], curr_crop.shape[1])
    p = cv2.resize(prev_crop[:h, :w], (64, 32))
    c = cv2.resize(curr_crop[:h, :w], (64, 32))
    flow = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(p, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(c, cv2.COLOR_RGB2GRAY),
        None, 0.5, 3, 15, 3, 5, 1.2, 0,
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
    return max(0.0, 1.0 - mag / 20.0)


def score_plate_candidate(
    sharpness: float,
    plate_size: float,
    detection_conf: float,
    brightness: float,
    contrast: float,
    viewing_angle: float,
    motion_penalty: float = 1.0,
) -> float:
    score = (
        0.30 * min(sharpness / 500.0, 1.0)
        + 0.25 * min(plate_size * 100, 1.0)
        + 0.20 * detection_conf
        + 0.10 * brightness
        + 0.10 * contrast
        + 0.05 * viewing_angle
    ) * motion_penalty
    return round(score, 4)
