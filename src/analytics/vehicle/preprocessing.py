import cv2
import numpy as np


def preprocess_for_ocr(crop: np.ndarray) -> list[np.ndarray]:
    variants = []

    if crop.size == 0:
        return variants

    variants.append(crop)

    h, w = crop.shape[:2]
    if max(h, w) < 100:
        scale = 200 / max(h, w)
        up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.append(up)

    sharpen = cv2.filter2D(crop, -1, np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]))
    variants.append(sharpen)

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    contrast = cv2.merge([l_eq, a, b])
    contrast = cv2.cvtColor(contrast, cv2.COLOR_LAB2BGR)
    variants.append(contrast)

    return variants
