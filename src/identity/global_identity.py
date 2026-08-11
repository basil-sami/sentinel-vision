"""Global identity data model.

A GlobalIdentity aggregates observations of one entity (person/vehicle)
across multiple cameras. The local tracker IDs within each camera are
temporary; this is the persistent cross-camera identity.
"""

import time
from dataclasses import dataclass, field


@dataclass
class Observation:
    """One sighting of an identity on one camera."""
    camera_id: str
    local_track_id: int
    class_name: str
    first_frame: int
    last_frame: int
    duration_frames: int
    matched_at: float = 0.0
    match_score: float = 0.0


@dataclass
class GlobalIdentity:
    id: int
    class_name: str
    embeddings: list = field(default_factory=list)   # np.ndarray, max EMBED_CAP
    observations: list[Observation] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0

    def add_observation(self, obs: Observation):
        self.observations.append(obs)
        if self.first_seen == 0.0 or obs.matched_at < self.first_seen:
            self.first_seen = obs.matched_at
        self.last_seen = max(self.last_seen, obs.matched_at)

    @property
    def cameras(self) -> list[str]:
        return sorted({o.camera_id for o in self.observations})

    @property
    def camera_count(self) -> int:
        return len(self.cameras)

    def to_dict(self, embeddings_b64: bool = True) -> dict:
        return {
            "id": self.id,
            "class_name": self.class_name,
            "camera_count": self.camera_count,
            "cameras": self.cameras,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "embeddings": len(self.embeddings),
            "observations": [
                {
                    "camera_id": o.camera_id,
                    "local_track_id": o.local_track_id,
                    "first_frame": o.first_frame,
                    "last_frame": o.last_frame,
                    "duration_frames": o.duration_frames,
                    "match_score": round(o.match_score, 3),
                }
                for o in self.observations
            ],
        }
