from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AttributeState(Enum):
    UNKNOWN = "unknown"
    PENDING = "pending"
    PROCESSING = "processing"
    VERIFIED = "verified"
    FAILED = "failed"
    LOCKED = "locked"


MAX_PLATE_ATTEMPTS = 5
COLOR_CONSISTENT_NEEDED = 3


@dataclass
class AttributeSlot:
    value: Any = None
    confidence: float = 0.0
    state: AttributeState = AttributeState.UNKNOWN
    attempts: int = 0
    observations: list = field(default_factory=list)


@dataclass
class TrackAttributes:
    plate: AttributeSlot = field(default_factory=AttributeSlot)
    color: AttributeSlot = field(default_factory=AttributeSlot)
    size_class: AttributeSlot = field(default_factory=AttributeSlot)
    vehicle_type: AttributeSlot = field(default_factory=AttributeSlot)

    def is_complete(self) -> bool:
        return all(
            s.state in (AttributeState.VERIFIED, AttributeState.LOCKED, AttributeState.FAILED)
            for s in (self.plate, self.color, self.size_class, self.vehicle_type)
        )

    def completeness_pct(self) -> float:
        total = 4
        done = sum(
            1 for s in (self.plate, self.color, self.size_class, self.vehicle_type)
            if s.state in (AttributeState.VERIFIED, AttributeState.LOCKED, AttributeState.FAILED)
        )
        return round(done / total * 100, 1)


class AttributeManager:
    def __init__(self):
        self._tracks: dict[int, TrackAttributes] = {}

    def get(self, track_id: int) -> TrackAttributes:
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackAttributes()
        return self._tracks[track_id]

    def remove(self, track_id: int):
        self._tracks.pop(track_id, None)

    def cleanup(self, active_ids: set[int]):
        stale = list(self._tracks.keys() - active_ids)
        for sid in stale:
            self._tracks.pop(sid, None)
