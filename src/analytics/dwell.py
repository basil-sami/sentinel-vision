class DwellTracker:
    def __init__(self):
        self._zone_entry: dict[tuple[int, str], int] = {}
        self._active: dict[tuple[int, str], int] = {}

    def update(self, track_id: int, zone_name: str, frame: int, in_zone: bool):
        key = (track_id, zone_name)
        if in_zone:
            if key not in self._active:
                self._active[key] = frame
                self._zone_entry[key] = frame
        else:
            self._active.pop(key, None)

    def current_dwell(self, track_id: int, zone_name: str, frame: int, fps: float = 25.0) -> float:
        entry_frame = self._active.get((track_id, zone_name))
        if entry_frame is None:
            return 0.0
        return (frame - entry_frame) / fps

    def total_dwell(self, track_id: int, zone_name: str) -> int:
        key = (track_id, zone_name)
        if key in self._active:
            entry = self._zone_entry[key]
            return self._active[key] - entry
        return 0

    def summary(self) -> dict:
        totals: dict[str, list[int]] = {}
        for (tid, zname), entry_frame in self._zone_entry.items():
            totals.setdefault(zname, []).append(0)
        for (tid, zname), active_since in self._active.items():
            entry = self._zone_entry.get((tid, zname), active_since)
            duration = active_since - entry
            totals.setdefault(zname, []).append(duration)
        result = {}
        for zname, durations in totals.items():
            result[zname] = {
                "total_dwell_frames": sum(durations),
                "average_dwell_frames": round(sum(durations) / len(durations), 1) if durations else 0,
                "max_dwell_frames": max(durations) if durations else 0,
                "unique_objects": len(durations),
            }
        return result
