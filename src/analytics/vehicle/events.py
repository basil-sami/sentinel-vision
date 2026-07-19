from src.models.event import Event


SPEEDING_THRESHOLD_MPS = 11.0


def speeding_event(track_id: int, speed_mps: float, location: list[int]) -> Event:
    return Event(
        event_type="speeding",
        track_id=track_id,
        class_name="vehicle",
        duration=speed_mps,
        location=location,
        message=f"Vehicle ID {track_id} speeding at {speed_mps:.1f} m/s ({speed_mps * 3.6:.0f} km/h)",
        severity="medium",
    )


def parking_event(track_id: int, plate: str, duration_sec: float, location: list[int]) -> Event:
    return Event(
        event_type="vehicle_parking",
        track_id=track_id,
        class_name="vehicle",
        duration=duration_sec,
        location=location,
        message=f"Vehicle {plate or f'ID {track_id}'} parked for {duration_sec:.0f}s",
        severity="info",
    )


def plate_read_event(track_id: int, plate: str, confidence: float, location: list[int]) -> Event:
    return Event(
        event_type="plate_read",
        track_id=track_id,
        class_name="vehicle",
        location=location,
        message=f"Plate {plate} (conf={confidence}) on vehicle ID {track_id}",
        severity="info",
        confidence=confidence,
    )


def repeat_visitor_event(plate: str, visit_count: int, track_id: int) -> Event:
    return Event(
        event_type="repeat_visitor",
        track_id=track_id,
        class_name="vehicle",
        message=f"Vehicle {plate} returning for visit #{visit_count}",
        severity="low",
    )
