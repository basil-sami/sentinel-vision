from src.models.event import Event


loiter_config: dict[str, int] = {
    "person": 600,
    "vehicle": 300,
}


class EventDetector:
    def __init__(self):
        self._loiter_warnings: dict[str, set[int]] = {}

    def check_loitering(
        self,
        track_id: int,
        class_name: str,
        zone_name: str,
        dwell_seconds: float,
        frame: int,
        cx: int,
        cy: int,
    ) -> Event | None:
        threshold = loiter_config.get(class_name, 600)
        if dwell_seconds < threshold:
            return None
        key = f"{zone_name}_{track_id}"
        warned = self._loiter_warnings.setdefault(key, set())
        warning_bucket = int(dwell_seconds / threshold)
        if warning_bucket in warned:
            return None
        warned.add(warning_bucket)
        return Event(
            event_type="possible_loitering",
            track_id=track_id,
            class_name=class_name,
            zone=zone_name,
            duration=dwell_seconds,
            location=[cx, cy],
            message=f"{class_name} ID {track_id} loitering in {zone_name} for {dwell_seconds:.0f}s",
        )

    def check_zone_entry(
        self,
        track_id: int,
        class_name: str,
        zone_name: str,
        cx: int,
        cy: int,
    ) -> Event:
        return Event(
            event_type="zone_entry",
            track_id=track_id,
            class_name=class_name,
            zone=zone_name,
            location=[cx, cy],
            message=f"{class_name} ID {track_id} entered {zone_name}",
        )

    def check_zone_exit(
        self,
        track_id: int,
        class_name: str,
        zone_name: str,
        dwell_seconds: float,
        cx: int,
        cy: int,
    ) -> Event:
        return Event(
            event_type="zone_exit",
            track_id=track_id,
            class_name=class_name,
            zone=zone_name,
            duration=dwell_seconds,
            location=[cx, cy],
            message=f"{class_name} ID {track_id} exited {zone_name} after {dwell_seconds:.0f}s",
        )
