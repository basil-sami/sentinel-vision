"""Shared PaddleOCR instance with thread-pool offloading.

Eliminates duplicate model loading (PlateDetector + PlateReader each
loaded their own instance) and runs OCR in a background thread so it
never blocks the main pipeline.
"""

import logging
import json
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

import numpy as np

log = logging.getLogger(__name__)


class OcrPool:
    """Single shared PaddleOCR instance + thread pool for offloading."""

    def __init__(self, max_workers: int = 1, device: str | None = None):
        self._lock = Lock()
        self._ocr = None
        self._ocr_fallback = False
        self._easyocr = None
        self._device = device
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._warmed_up = False

    def _get_ocr(self):
        if self._ocr is not None:
            return self._ocr
        if self._ocr_fallback:
            return None
        with self._lock:
            if self._ocr is not None:
                return self._ocr
            try:
                from paddleocr import PaddleOCR
                # PaddleOCR 3.x uses predict() and renamed/removed most 2.x
                # constructor flags.  Keep the old constructor as a fallback
                # for Colab sessions that already have PaddleOCR 2.x cached.
                device = self._device or "cpu"
                try:
                    self._ocr = PaddleOCR(
                        lang="en",
                        device=device,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                    )
                except (TypeError, ValueError):
                    legacy_kwargs = {"use_angle_cls": False, "lang": "en", "show_log": False}
                    try:
                        self._ocr = PaddleOCR(**legacy_kwargs, device=device)
                    except TypeError:
                        self._ocr = PaddleOCR(**legacy_kwargs)
            except (ImportError, ModuleNotFoundError):
                self._ocr_fallback = True
                return None
            except Exception as exc:
                log.warning("PaddleOCR initialization failed: %s", exc)
                self._ocr_fallback = True
                return None
        return self._ocr

    def _get_easyocr(self):
        if self._easyocr is not None:
            return self._easyocr
        try:
            import easyocr
            use_gpu = bool(self._device and (self._device.startswith("cuda") or self._device.startswith("gpu")))
            self._easyocr = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        except Exception as exc:
            log.warning("EasyOCR fallback unavailable: %s", exc)
            self._easyocr = False
        return self._easyocr if self._easyocr is not False else None

    def warmup(self):
        """Pre-load PaddleOCR on a tiny dummy image during init."""
        if self._warmed_up:
            return
        ocr = self._get_ocr()
        if ocr is None:
            return
        dummy = np.zeros((64, 128, 3), dtype=np.uint8)
        self._infer(dummy, detect=True, recognize=True)
        self._warmed_up = True
        log.info("OcrPool: PaddleOCR warmed up")

    @staticmethod
    def _as_dict(result):
        """Convert PaddleOCR 3.x result objects into plain dictionaries."""
        if isinstance(result, dict):
            return result.get("res", result)
        value = getattr(result, "json", None)
        if callable(value):
            value = value()
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = None
        if isinstance(value, dict):
            return value.get("res", value)
        value = getattr(result, "to_dict", None)
        if callable(value):
            value = value()
            if isinstance(value, dict):
                return value.get("res", value)
        return None

    def _infer(self, crop: np.ndarray, detect: bool, recognize: bool):
        ocr = self._get_ocr()
        if ocr is None:
            return None
        try:
            if hasattr(ocr, "predict"):
                return list(ocr.predict(crop))
            return ocr.ocr(crop, det=detect, rec=recognize, cls=False)
        except Exception as exc:
            log.debug("PaddleOCR inference failed: %s", exc)
            return None

    @staticmethod
    def _polygon_bbox(poly) -> tuple[int, int, int, int] | None:
        try:
            array = np.asarray(poly)
            if array.size == 4 and array.ndim == 1:
                x1, y1, x2, y2 = array.tolist()
                return int(x1), int(y1), int(x2), int(y2)
            points = array.reshape(-1, 2)
            if len(points) < 2:
                return None
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            return int(x1), int(y1), int(x2), int(y2)
        except (TypeError, ValueError):
            return None

    def _legacy_lines(self, result):
        """Yield (polygon, text, confidence) from PaddleOCR 2.x output."""
        if not isinstance(result, list):
            return
        items = result[0] if len(result) == 1 and isinstance(result[0], list) else result
        for item in items or []:
            if not isinstance(item, (list, tuple, np.ndarray)) or not len(item):
                continue
            # det=True, rec=False in PaddleOCR 2.x returns bare polygons.
            if len(item) >= 4 and all(isinstance(point, (list, tuple, np.ndarray)) for point in item[:4]):
                bbox = self._polygon_bbox(item)
                if bbox is not None:
                    yield bbox, "", 0.8
                continue
            poly = item[0]
            bbox = self._polygon_bbox(poly)
            if bbox is None:
                continue
            text, confidence = "", 0.0
            if len(item) > 1 and isinstance(item[1], (list, tuple)):
                text = str(item[1][0]) if item[1] else ""
                confidence = float(item[1][1]) if len(item[1]) > 1 else 0.0
            yield bbox, text, confidence

    def detect_sync(self, crop: np.ndarray) -> list | None:
        """Return normalized plate/text boxes for PaddleOCR 2.x or 3.x."""
        results = self._infer(crop, detect=True, recognize=False)
        if not results:
            reader = self._get_easyocr()
            if reader is None:
                return None
            try:
                results = reader.readtext(crop, detail=1)
                return [
                    {"bbox": self._polygon_bbox(item[0]), "confidence": float(item[2])}
                    for item in results
                    if len(item) >= 3 and self._polygon_bbox(item[0]) is not None
                ] or None
            except Exception:
                return None
        output = []
        modern = self._as_dict(results[0]) if isinstance(results, list) else self._as_dict(results)
        if modern:
            polys = modern.get("dt_polys", [])
            scores = modern.get("dt_scores", [])
            for index, poly in enumerate(polys):
                bbox = self._polygon_bbox(poly)
                if bbox:
                    output.append({"bbox": bbox, "confidence": float(scores[index]) if index < len(scores) else 0.8})
            return output or None
        for bbox, _, confidence in self._legacy_lines(results):
            output.append({"bbox": bbox, "confidence": confidence or 0.8})
        return output or None

    def read_sync(self, crop: np.ndarray) -> list | None:
        """Return normalized text candidates for PaddleOCR 2.x or 3.x."""
        results = self._infer(crop, detect=True, recognize=True)
        if not results:
            reader = self._get_easyocr()
            if reader is None:
                return None
            try:
                results = reader.readtext(crop, detail=1)
                return [
                    {"bbox": self._polygon_bbox(item[0]), "text": str(item[1]), "confidence": float(item[2])}
                    for item in results
                    if len(item) >= 3 and self._polygon_bbox(item[0]) is not None
                ] or None
            except Exception:
                return None
        output = []
        modern = self._as_dict(results[0]) if isinstance(results, list) else self._as_dict(results)
        if modern:
            boxes = modern.get("rec_boxes", modern.get("rec_polys", []))
            texts = modern.get("rec_texts", [])
            scores = modern.get("rec_scores", [])
            for index, box in enumerate(boxes):
                bbox = self._polygon_bbox(box)
                if bbox and index < len(texts):
                    output.append({
                        "bbox": bbox,
                        "text": str(texts[index]),
                        "confidence": float(scores[index]) if index < len(scores) else 0.0,
                    })
            return output or None
        for bbox, text, confidence in self._legacy_lines(results):
            output.append({"bbox": bbox, "text": text, "confidence": confidence})
        return output or None

    def submit_detect(self, crop: np.ndarray) -> Future:
        """Submit plate-detection OCR to thread pool."""
        return self._executor.submit(self.detect_sync, crop)

    def submit_read(self, crop: np.ndarray) -> Future:
        """Submit plate-reading OCR to thread pool."""
        return self._executor.submit(self.read_sync, crop)


_global_pools: dict[str, OcrPool] = {}
_pool_lock = Lock()


def get_ocr_pool(max_workers: int = 1, device: str | None = None) -> OcrPool:
    key = device or "cpu"
    if key not in _global_pools:
        with _pool_lock:
            if key not in _global_pools:
                _global_pools[key] = OcrPool(max_workers=max_workers, device=device)
    return _global_pools[key]
