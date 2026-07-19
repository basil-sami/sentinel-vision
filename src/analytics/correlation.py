import time
from dataclasses import dataclass, field
from typing import Any


INCIDENT_RULES = [
    {
        "name": "suspicious_activity",
        "severity": "critical",
        "time_window": 120,
        "required": ["zone_entry"],
        "optional": ["possible_loitering", "abandoned_object", "rapid_movement"],
        "min_optional": 2,
        "description": "Person enters zone, loiters, abandons object, flees",
    },
    {
        "name": "prolonged_loitering",
        "severity": "high",
        "time_window": 300,
        "required": ["possible_loitering"],
        "optional": ["object_interaction"],
        "min_optional": 1,
        "description": "Person loiters with object interaction",
    },
    {
        "name": "gate_breach",
        "severity": "high",
        "time_window": 60,
        "required": ["gate_crossing"],
        "optional": ["rapid_movement"],
        "min_optional": 0,
        "description": "Unauthorized gate crossing",
    },
    {
        "name": "object_drop",
        "severity": "high",
        "time_window": 30,
        "required": ["abandoned_object"],
        "optional": ["object_interaction"],
        "min_optional": 1,
        "description": "Object abandoned after person interaction",
    },
]


@dataclass
class Incident:
    incident_type: str
    severity: str
    events: list = field(default_factory=list)
    track_ids: set = field(default_factory=set)
    start_time: float = field(default_factory=time.time)
    last_time: float = field(default_factory=time.time)
    summary: str = ""
    status: str = "open"

    def to_dict(self) -> dict:
        return {
            "incident_type": self.incident_type,
            "severity": self.severity,
            "track_ids": list(self.track_ids),
            "event_count": len(self.events),
            "start_time": self.start_time,
            "last_time": self.last_time,
            "summary": self.summary,
            "status": self.status,
        }


class EventCorrelator:
    def __init__(self):
        self._incidents: list[Incident] = []
        self._track_events: dict[int, list[dict]] = {}

    def process_event(self, event: dict) -> Incident | None:
        track_id = event.get("track_id")
        if track_id is None or track_id < 0:
            return None

        self._track_events.setdefault(track_id, []).append(event)
        recent = self._track_events[track_id][-20:]

        for rule in INCIDENT_RULES:
            incident = self._check_rule(rule, recent, track_id)
            if incident:
                self._incidents.append(incident)
                return incident
        return None

    def _check_rule(self, rule: dict, events: list[dict], track_id: int) -> Incident | None:
        time_window = rule["time_window"]
        cutoff = time.time() - time_window
        window = [e for e in events if e.get("timestamp", 0) > cutoff]

        event_types = [e.get("type", "") for e in window]

        has_required = all(req in event_types for req in rule["required"])
        if not has_required:
            return None

        optional_matched = sum(1 for opt in rule["optional"] if opt in event_types)
        if optional_matched < rule["min_optional"]:
            return None

        summary = (
            f"{rule['description']}: "
            f"track {track_id}: "
            + ", ".join(
                sorted(set(e.get("type", "") for e in window if e.get("type", "") in rule["required"] + rule["optional"]))
            )
        )

        return Incident(
            incident_type=rule["name"],
            severity=rule["severity"],
            events=list(window),
            track_ids={track_id},
            summary=summary,
        )

    def incidents(self, status: str | None = None) -> list[Incident]:
        if status:
            return [i for i in self._incidents if i.status == status]
        return list(self._incidents)

    def open_incidents(self) -> list[Incident]:
        return self.incidents("open")

    def close_incident(self, incident: Incident):
        incident.status = "closed"
