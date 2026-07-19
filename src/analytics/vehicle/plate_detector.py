import cv2
import numpy as np


class PlateDetector:
    def __init__(self):
        self._ocr = None

    def _lazy_init(self):
        if self._ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        except ImportError:
            self._ocr = False

    def detect(self, frame: np.ndarray, vehicle_bbox: tuple[int, int, int, int]) -> dict | None:
        self._lazy_init()

        x1, y1, x2, y2 = vehicle_bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < 20 or crop.shape[1] < 20:
            return None

        if self._ocr is False:
            return self._contour_fallback(crop, x1, y1)

        if self._ocr is not None:
            try:
                results = self._ocr.ocr(crop, det=True, rec=False, cls=False)
            except Exception:
                results = None

            if results and len(results) > 0 and results[0] is not None:
                best = max(results[0], key=lambda r: (r[1][2] - r[1][0]) * (r[1][3] - r[1][1]))
                poly = best[0]
                xs = [int(p[0]) for p in poly]
                ys = [int(p[1]) for p in poly]
                bx1, bx2 = min(xs), max(xs)
                by1, by2 = min(ys), max(ys)
                return {
                    "bbox": (x1 + bx1, y1 + by1, x1 + bx2, y1 + by2),
                    "confidence": 0.8,
                    "method": "paddle",
                }

        return self._contour_fallback(crop, x1, y1)

    def _contour_fallback(self, crop: np.ndarray, offset_x: int, offset_y: int) -> dict | None:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        best = None
        best_score = 0
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect = w / h if h > 0 else 0
                area = w * h
                crop_area = crop.shape[0] * crop.shape[1]
                if 1.5 < aspect < 6.0 and 0.02 < area / crop_area < 0.6:
                    score = area
                    if score > best_score:
                        best_score = score
                        best = (x, y, x + w, y + h)

        if best:
            return {
                "bbox": (offset_x + best[0], offset_y + best[1], offset_x + best[2], offset_y + best[3]),
                "confidence": min(best_score / 10000.0, 0.6),
                "method": "contour",
            }
        return None
