from src.analytics.scene.carrying import CarryingDetector
from src.analytics.scene.overloaded_vehicle import OverloadedVehicleDetector
from src.models.event import Event


class SceneAnalyzer:
    def __init__(
        self,
        carrying_config: dict | None = None,
        overloaded_vehicle_config: dict | None = None,
    ):
        cfg1 = carrying_config or {}
        cfg2 = overloaded_vehicle_config or {}
        self.carrying = CarryingDetector(
            min_overlap_iou=cfg1.get("min_overlap_iou", 0.05),
            min_co_motion_frames=cfg1.get("min_co_motion_frames", 5),
            speed_similarity=cfg1.get("speed_similarity", 0.5),
        )
        self.overloaded = OverloadedVehicleDetector(
            min_persons=cfg2.get("min_persons", 3),
            min_co_motion_frames=cfg2.get("min_co_motion_frames", 5),
            speed_similarity=cfg2.get("speed_similarity", 0.4),
        )

    def process_frame(
        self,
        tracks: list,
        frame_index: int,
        calibrator=None,
        zone_mgr=None,
    ) -> list[Event]:
        events: list[Event] = []

        events.extend(self.carrying.process(tracks, frame_index))
        events.extend(self.overloaded.process(tracks, frame_index))

        return events
