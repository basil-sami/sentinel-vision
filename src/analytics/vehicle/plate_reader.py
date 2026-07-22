from concurrent.futures import Future

import numpy as np

from src.analytics.vehicle.preprocessing import preprocess_for_ocr
from src.analytics.vehicle.validation import validate_plate
from src.analytics.vehicle.ocr_pool import get_ocr_pool


class PlateReader:
    def __init__(self, ocr_pool=None):
        self._ocr_pool = ocr_pool or get_ocr_pool()
        self._pending: dict[int, Future] = {}

    def read(self, plate_crop: np.ndarray) -> dict:
        variants = preprocess_for_ocr(plate_crop)
        if not variants:
            return {"plate": "", "confidence": 0.0}

        best_text = ""
        best_conf = 0.0

        for variant in variants:
            results = self._run_ocr(variant)
            if results is None:
                continue
            if isinstance(results, list) and len(results) > 0 and results[0] is not None:
                for line in results[0]:
                    bbox, (text, conf) = line
                    validated, qual = validate_plate(text)
                    if validated and conf > best_conf:
                        w = bbox[1][0] - bbox[0][0]
                        h = bbox[1][1] - bbox[0][1]
                        aspect = w / h if h > 0 else 0
                        if 1.5 < aspect < 8.0 or len(validated) >= 3:
                            best_text = validated
                            best_conf = conf * qual

        return {"plate": best_text, "confidence": round(best_conf, 3)}

    def read_async(self, plate_crop: np.ndarray) -> Future:
        """Submit OCR read to thread pool, return Future."""
        future = self._ocr_pool.submit_read(plate_crop)
        return future

    def collect_read(self, future: Future) -> dict:
        """Collect async OCR result from a Future."""
        try:
            results = future.result(timeout=10)
        except Exception:
            return {"plate": "", "confidence": 0.0}
        if results is None:
            return {"plate": "", "confidence": 0.0}

        best_text = ""
        best_conf = 0.0

        if isinstance(results, list) and len(results) > 0 and results[0] is not None:
            for line in results[0]:
                bbox, (text, conf) = line
                validated, qual = validate_plate(text)
                if validated and conf > best_conf:
                    w = bbox[1][0] - bbox[0][0]
                    h = bbox[1][1] - bbox[0][1]
                    aspect = w / h if h > 0 else 0
                    if 1.5 < aspect < 8.0 or len(validated) >= 3:
                        best_text = validated
                        best_conf = conf * qual

        return {"plate": best_text, "confidence": round(best_conf, 3)}

    def _run_ocr(self, variant: np.ndarray) -> list | None:
        return self._ocr_pool.read_sync(variant)

    def _easyocr_fallback(self, plate_crop) -> dict:
        try:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            results = reader.readtext(plate_crop)
            if not results:
                return {"plate": "", "confidence": 0.0}
            best = max(results, key=lambda r: r[2])
            text, conf = best[1], best[2]
            validated, qual = validate_plate(text)
            return {"plate": validated, "confidence": round(float(conf) * qual, 3)}
        except ImportError:
            return {"plate": "", "confidence": 0.0, "error": "no OCR available"}
