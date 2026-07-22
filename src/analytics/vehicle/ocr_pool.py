"""Shared PaddleOCR instance with thread-pool offloading.

Eliminates duplicate model loading (PlateDetector + PlateReader each
loaded their own instance) and runs OCR in a background thread so it
never blocks the main pipeline.
"""

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

import numpy as np

log = logging.getLogger(__name__)


class OcrPool:
    """Single shared PaddleOCR instance + thread pool for offloading."""

    def __init__(self, max_workers: int = 1):
        self._lock = Lock()
        self._ocr = None
        self._ocr_fallback = False
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
                self._ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
            except ImportError:
                self._ocr_fallback = True
                return None
        return self._ocr

    def warmup(self):
        """Pre-load PaddleOCR on a tiny dummy image during init."""
        if self._warmed_up:
            return
        ocr = self._get_ocr()
        if ocr is None:
            return
        dummy = np.zeros((64, 128, 3), dtype=np.uint8)
        ocr.ocr(dummy, det=True, rec=True, cls=False)
        self._warmed_up = True
        log.info("OcrPool: PaddleOCR warmed up")

    def detect_sync(self, crop: np.ndarray) -> list | None:
        """Run plate detection synchronously (shares model, no thread hop)."""
        ocr = self._get_ocr()
        if ocr is None:
            return None
        try:
            return ocr.ocr(crop, det=True, rec=False, cls=False)
        except Exception:
            return None

    def read_sync(self, crop: np.ndarray) -> list | None:
        """Run plate reading synchronously (shares model, no thread hop)."""
        ocr = self._get_ocr()
        if ocr is None:
            return None
        try:
            return ocr.ocr(crop, det=True, rec=True, cls=False)
        except Exception:
            return None

    def submit_detect(self, crop: np.ndarray) -> Future:
        """Submit plate-detection OCR to thread pool."""
        return self._executor.submit(self.detect_sync, crop)

    def submit_read(self, crop: np.ndarray) -> Future:
        """Submit plate-reading OCR to thread pool."""
        return self._executor.submit(self.read_sync, crop)


_global_pool = None
_pool_lock = Lock()


def get_ocr_pool(max_workers: int = 1) -> OcrPool:
    global _global_pool
    if _global_pool is None:
        with _pool_lock:
            if _global_pool is None:
                _global_pool = OcrPool(max_workers=max_workers)
    return _global_pool
