from dataclasses import dataclass
import numpy as np

from .frame_quality import (
    compute_sharpness,
    compute_plate_size,
    compute_viewing_angle,
    compute_brightness_contrast,
    score_plate_candidate,
)


@dataclass
class PlateCandidate:
    score: float
    frame_index: int
    plate_crop: np.ndarray | None
    vehicle_crop: np.ndarray | None
    plate_bbox: tuple[int, int, int, int]
    sharpness: float
    plate_size_ratio: float

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "frame": self.frame_index,
            "sharpness": round(self.sharpness, 1),
            "plate_size_ratio": round(self.plate_size_ratio, 4),
        }


class TopKBuffer:
    def __init__(self, k: int = 5):
        self._k = k
        self._candidates: list[PlateCandidate] = []

    def add(self, candidate: PlateCandidate):
        self._candidates.append(candidate)
        self._candidates.sort(key=lambda c: c.score, reverse=True)
        if len(self._candidates) > self._k:
            self._candidates = self._candidates[:self._k]

    def best(self) -> PlateCandidate | None:
        return self._candidates[0] if self._candidates else None

    def top_k(self) -> list[PlateCandidate]:
        return list(self._candidates)

    def clear(self):
        self._candidates.clear()

    def __len__(self) -> int:
        return len(self._candidates)

    def evaluate_and_add(
        self,
        plate_crop: np.ndarray,
        vehicle_crop: np.ndarray,
        plate_bbox: tuple[int, int, int, int],
        detection_conf: float,
        frame_index: int,
        frame_area: int,
    ):
        sharpness = compute_sharpness(plate_crop)
        plate_size = compute_plate_size(plate_bbox, frame_area)
        brightness, contrast = compute_brightness_contrast(plate_crop)
        viewing_angle = compute_viewing_angle(plate_bbox)
        score = score_plate_candidate(sharpness, plate_size, detection_conf, brightness, contrast, viewing_angle)

        self.add(PlateCandidate(
            score=score,
            frame_index=frame_index,
            plate_crop=plate_crop,
            vehicle_crop=vehicle_crop,
            plate_bbox=plate_bbox,
            sharpness=sharpness,
            plate_size_ratio=plate_size,
        ))
