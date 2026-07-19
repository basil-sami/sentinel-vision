from src.analytics.vehicle.preprocessing import preprocess_for_ocr
from src.analytics.vehicle.validation import validate_plate


class PlateReader:
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

    def read(self, plate_crop) -> dict:
        self._lazy_init()
        if self._ocr is False:
            return self._easyocr_fallback(plate_crop)

        variants = preprocess_for_ocr(plate_crop)
        if not variants:
            return {"plate": "", "confidence": 0.0}

        best_text = ""
        best_conf = 0.0

        for variant in variants:
            try:
                results = self._ocr.ocr(variant, det=True, rec=True, cls=False)
            except Exception:
                continue

            if not results or len(results) == 0 or results[0] is None:
                continue

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
