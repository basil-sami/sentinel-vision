import numpy as np

from src.models.event import Event


class AbandonedDetector:
    def __init__(self, max_association_distance: float = 80.0, stationary_threshold_frames: int = 300):
        self._max_association_distance = max_association_distance
        self._stationary_threshold_frames = stationary_threshold_frames
        self._owner_links: dict[int, int] = {}
        self._stationary_since: dict[int, int] = {}
        self._reported: set[int] = set()

    def update(
        self,
        track_id: int,
        class_name: str,
        bbox: tuple[int, int, int, int],
        frame: int,
        active_tracks: list,
    ) -> Event | None:
        if class_name == "person":
            return None

        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        owners = [t for t in active_tracks if t.class_name == "person"]

        if owners:
            nearest = min(
                owners,
                key=lambda o: np.linalg.norm(
                    np.array(cx) - np.array((o.bbox[0] + o.bbox[2]) // 2)
                ),
            )
            nearest_cx = (nearest.bbox[0] + nearest.bbox[2]) // 2
            nearest_cy = (nearest.bbox[1] + nearest.bbox[3]) // 2
            dist = np.linalg.norm(np.array([cx, cy]) - np.array([nearest_cx, nearest_cy]))

            if dist < self._max_association_distance:
                self._owner_links[track_id] = nearest.id
                self._stationary_since.pop(track_id, None)
                return None
            else:
                self._owner_links.pop(track_id, None)
                if track_id not in self._stationary_since:
                    self._stationary_since[track_id] = frame

                duration = frame - self._stationary_since[track_id]
                if duration >= self._stationary_threshold_frames and track_id not in self._reported:
                    self._reported.add(track_id)
                    return Event(
                        event_type="abandoned_object",
                        track_id=track_id,
                        class_name=class_name,
                        duration=duration / 25.0,
                        location=[cx, cy],
                        message=f"Abandoned {class_name} ID {track_id} (owner ID {self._owner_links.get(track_id, 'unknown')})",
                    )
        else:
            self._owner_links.pop(track_id, None)
            if track_id not in self._stationary_since:
                self._stationary_since[track_id] = frame
            duration = frame - self._stationary_since[track_id]
            if duration >= self._stationary_threshold_frames and track_id not in self._reported:
                self._reported.add(track_id)
                return Event(
                    event_type="abandoned_object",
                    track_id=track_id,
                    class_name=class_name,
                    duration=duration / 25.0,
                    location=[cx, cy],
                    message=f"Abandoned {class_name} ID {track_id}",
                )

        return None
