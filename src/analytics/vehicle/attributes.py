import cv2
import numpy as np


_COLOR_NAMES = {
    "black": ([0, 0, 0], [50, 50, 80]),
    "white": ([200, 200, 200], [255, 255, 255]),
    "gray": ([80, 80, 80], [200, 200, 200]),
    "silver": ([160, 160, 170], [220, 220, 230]),
    "red": ([0, 0, 100], [80, 80, 255]),
    "blue": ([80, 0, 0], [255, 100, 80]),
    "green": ([0, 80, 0], [80, 255, 80]),
    "yellow": ([0, 150, 150], [100, 255, 255]),
    "orange": ([0, 100, 200], [80, 180, 255]),
    "brown": ([0, 50, 100], [80, 120, 180]),
}


def extract_vehicle_color(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> dict:
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return {"color": "unknown", "confidence": 0.0}

    h, w = crop.shape[:2]
    center_region = crop[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    if center_region.size == 0:
        center_region = crop

    pixels = center_region.reshape(-1, 3)
    pixels = pixels.astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant = centers[np.argmax(np.bincount(labels.flatten()))].astype(int)

    best_color = "unknown"
    best_dist = float("inf")
    for name, (lower, upper) in _COLOR_NAMES.items():
        avg = [(lower[i] + upper[i]) // 2 for i in range(3)]
        dist = sum((dominant[i] - avg[i]) ** 2 for i in range(3)) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_color = name

    confidence = max(0.0, 1.0 - best_dist / 400.0)
    return {"color": best_color, "confidence": round(confidence, 3), "rgb": dominant.tolist()}


def vehicle_size_class(bbox: tuple[int, int, int, int]) -> str:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    area = w * h
    if area < 5000:
        return "small"
    elif area < 20000:
        return "medium"
    return "large"
