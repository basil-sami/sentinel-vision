import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VehicleRecord:
    plate: str
    color: str
    vehicle_type: str
    size_class: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    track_ids: set[int] = field(default_factory=set)
    visit_count: int = 1
    total_duration_sec: float = 0.0
    current_parking_start: float | None = None

    def to_dict(self) -> dict:
        return {
            "plate": self.plate,
            "color": self.color,
            "vehicle_type": self.vehicle_type,
            "size_class": self.size_class,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "track_ids": list(self.track_ids),
            "visit_count": self.visit_count,
            "total_duration_sec": round(self.total_duration_sec, 1),
        }


class VehicleRegistry:
    def __init__(self, parking_timeout_sec: float = 300.0):
        self._vehicles: dict[str, VehicleRecord] = {}
        self._track_to_plate: dict[int, str] = {}
        self._plate_to_tracks: dict[str, set[int]] = {}
        self._parking_timeout = parking_timeout_sec

    def register(
        self,
        track_id: int,
        plate: str,
        color: str,
        vehicle_type: str,
        size_class: str,
    ) -> VehicleRecord:
        if plate:
            if plate not in self._vehicles:
                self._vehicles[plate] = VehicleRecord(
                    plate=plate,
                    color=color,
                    vehicle_type=vehicle_type,
                    size_class=size_class,
                )
            rec = self._vehicles[plate]
            rec.last_seen = time.time()
            rec.track_ids.add(track_id)
            self._plate_to_tracks.setdefault(plate, set()).add(track_id)

            if rec.current_parking_start is not None:
                duration = time.time() - rec.current_parking_start
                rec.total_duration_sec += duration
                rec.current_parking_start = None

            self._track_to_plate[track_id] = plate
            return rec

        vehicle_key = f"v_{track_id}_{color}_{vehicle_type}"
        if vehicle_key not in self._vehicles:
            self._vehicles[vehicle_key] = VehicleRecord(
                plate="",
                color=color,
                vehicle_type=vehicle_type,
                size_class=size_class,
            )
        rec = self._vehicles[vehicle_key]
        rec.last_seen = time.time()
        rec.track_ids.add(track_id)
        return rec

    def mark_parking(self, track_id: int):
        plate = self._track_to_plate.get(track_id)
        if plate and plate in self._vehicles:
            self._vehicles[plate].current_parking_start = time.time()

    def mark_departure(self, track_id: int):
        plate = self._track_to_plate.get(track_id)
        if plate and plate in self._vehicles:
            rec = self._vehicles[plate]
            if rec.current_parking_start is not None:
                duration = time.time() - rec.current_parking_start
                rec.total_duration_sec += duration
                rec.current_parking_start = None
            rec.visit_count += 1

    def get_by_plate(self, plate: str) -> VehicleRecord | None:
        return self._vehicles.get(plate)

    def get_by_track(self, track_id: int) -> VehicleRecord | None:
        plate = self._track_to_plate.get(track_id)
        if plate:
            return self._vehicles.get(plate)
        return None

    def all(self) -> list[VehicleRecord]:
        return list(self._vehicles.values())

    def summary(self) -> dict[str, Any]:
        return {
            "total_vehicles": len(self._vehicles),
            "with_plates": sum(1 for v in self._vehicles.values() if v.plate),
            "without_plates": sum(1 for v in self._vehicles.values() if not v.plate),
            "total_visits": sum(v.visit_count for v in self._vehicles.values()),
        }
