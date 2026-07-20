from dataclasses import dataclass, field
import json
from pathlib import Path


SEVERITY_MAP = {
    "gate_crossing": "info",
    "zone_entry": "info",
    "zone_exit": "info",
    "object_interaction": "info",
    "group_traveling": "low",
    "possible_loitering": "medium",
    "wrong_direction": "medium",
    "rapid_movement": "medium",
    "abandoned_object": "high",
    "crowd_forming": "medium",
    "camera_failure": "critical",
    "person_carrying": "medium",
    "overloaded_vehicle": "high",
}


def severity_for(event_type: str) -> str:
    return SEVERITY_MAP.get(event_type, "info")


@dataclass
class Event:
    event_type: str
    track_id: int
    class_name: str = ""
    zone: str = ""
    duration: float = 0.0
    location: list[int] | None = None
    message: str = ""
    severity: str = ""
    confidence: float = 1.0

    def __post_init__(self):
        if not self.severity:
            self.severity = severity_for(self.event_type)

    def to_dict(self) -> dict:
        d = {
            "type": self.event_type,
            "track_id": self.track_id,
            "class": self.class_name,
            "severity": self.severity,
            "confidence": round(self.confidence, 2),
        }
        if self.zone:
            d["zone"] = self.zone
        if self.duration:
            d["duration"] = round(self.duration, 1)
        if self.location:
            d["location"] = self.location
        if self.message:
            d["message"] = self.message
        return d


class EventStore:
    def __init__(self):
        self._events: list[Event] = []

    def add(self, event: Event):
        self._events.append(event)

    def all(self) -> list[Event]:
        return list(self._events)

    def by_type(self, event_type: str) -> list[Event]:
        return [e for e in self._events if e.event_type == event_type]

    def by_severity(self, level: str) -> list[Event]:
        return [e for e in self._events if e.severity == level]

    def critical(self) -> list[Event]:
        return self.by_severity("critical")

    def export(self) -> list[dict]:
        return [e.to_dict() for e in self._events]

    def save(self, path: str | Path):
        path = Path(path)
        path.write_text(json.dumps(self.export(), indent=2))
