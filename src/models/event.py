from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class Event:
    event_type: str
    track_id: int
    class_name: str = ""
    zone: str = ""
    duration: float = 0.0
    location: list[int] | None = None
    message: str = ""

    def to_dict(self) -> dict:
        d = {
            "type": self.event_type,
            "track_id": self.track_id,
            "class": self.class_name,
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

    def export(self) -> list[dict]:
        return [e.to_dict() for e in self._events]

    def save(self, path: str | Path):
        path = Path(path)
        path.write_text(json.dumps(self.export(), indent=2))
