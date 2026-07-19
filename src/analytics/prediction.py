import numpy as np


class TrackPredictor:
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
        self._kf: dict[int, dict] = {}
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        self._predicted: dict[int, tuple[float, float, float]] = {}

    def update(self, track_id: int, cx: float, cy: float, frame: int):
        if track_id not in self._kf:
            kf = {
                "x": np.array([cx, cy, 0.0, 0.0], dtype=np.float32),
                "P": np.eye(4, dtype=np.float32) * 10.0,
                "F": np.array([
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ], dtype=np.float32),
                "H": np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                ], dtype=np.float32),
                "Q": np.eye(4, dtype=np.float32) * self._process_noise,
                "R": np.eye(2, dtype=np.float32) * self._measurement_noise,
                "last_frame": frame,
            }
            self._kf[track_id] = kf
            self._predicted[track_id] = (cx, cy, 1.0)
            return

        kf = self._kf[track_id]
        dt = frame - kf["last_frame"]
        if dt > 1:
            kf["F"][0, 2] = dt
            kf["F"][1, 3] = dt

        kf["x"] = kf["F"] @ kf["x"]
        kf["P"] = kf["F"] @ kf["P"] @ kf["F"].T + kf["Q"]

        z = np.array([cx, cy], dtype=np.float32)
        y = z - kf["H"] @ kf["x"]
        S = kf["H"] @ kf["P"] @ kf["H"].T + kf["R"]
        K = kf["P"] @ kf["H"].T @ np.linalg.inv(S)

        kf["x"] = kf["x"] + K @ y
        kf["P"] = (np.eye(4) - K @ kf["H"]) @ kf["P"]
        kf["last_frame"] = frame

        confidence = 1.0 - float(np.trace(kf["P"][:2, :2]) / 40.0)
        confidence = max(0.1, min(1.0, confidence))
        self._predicted[track_id] = (float(kf["x"][0]), float(kf["x"][1]), confidence)

    def predict(self, track_id: int) -> tuple[float, float, float] | None:
        return self._predicted.get(track_id)

    def predict_all(self) -> dict[int, tuple[float, float, float]]:
        return dict(self._predicted)

    def remove(self, track_id: int):
        self._kf.pop(track_id, None)
        self._predicted.pop(track_id, None)
