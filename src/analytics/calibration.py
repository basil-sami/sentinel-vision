import numpy as np
import cv2


class Calibrator:
    def __init__(self):
        self._H: np.ndarray | None = None
        self._H_inv: np.ndarray | None = None
        self._image_points: list[list[int]] = []
        self._world_points: list[list[float]] = []

    def add_point(self, image_xy: list[int], world_xy: list[float]):
        self._image_points.append(image_xy)
        self._world_points.append(world_xy)

    def add_points(self, correspondences: list[dict]):
        for c in correspondences:
            self.add_point(c["image"], c["world"])

    def compute(self) -> bool:
        if len(self._image_points) < 4:
            return False
        src = np.array(self._image_points, dtype=np.float32)
        dst = np.array(self._world_points, dtype=np.float32)
        self._H, _ = cv2.findHomography(src, dst)
        if self._H is not None:
            self._H_inv = np.linalg.inv(self._H)
            return True
        return False

    @property
    def is_calibrated(self) -> bool:
        return self._H is not None

    def image_to_world(self, x: float, y: float) -> tuple[float, float]:
        if self._H is None:
            return (float(x), float(y))
        pt = np.array([[[x, y]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, self._H)
        return (float(world[0, 0, 0]), float(world[0, 0, 1]))

    def world_distance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def image_distance_in_world(self, x1: float, y1: float, x2: float, y2: float) -> float:
        w1 = self.image_to_world(x1, y1)
        w2 = self.image_to_world(x2, y2)
        return self.world_distance(w1, w2)

    def path_length_in_world(self, path: list[list[int]]) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(path)):
            total += self.image_distance_in_world(
                path[i - 1][0], path[i - 1][1],
                path[i][0], path[i][1]
            )
        return round(total, 2)

    def speed_in_world(self, path: list[list[int]], fps: float = 25.0) -> float:
        if len(path) < 2:
            return 0.0
        distance = self.path_length_in_world(path)
        duration = len(path) / fps
        return round(distance / duration, 2) if duration > 0 else 0.0

    def get_config(self) -> dict:
        return {
            "image_points": self._image_points,
            "world_points": self._world_points,
            "calibrated": self.is_calibrated,
        }

    @classmethod
    def from_config(cls, config: dict) -> "Calibrator":
        cal = cls()
        img_pts = config.get("image_points", [])
        wld_pts = config.get("world_points", [])
        for ip, wp in zip(img_pts, wld_pts):
            cal.add_point(ip, wp)
        cal.compute()
        return cal
