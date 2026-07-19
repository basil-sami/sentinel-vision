import numpy as np

from src.models.event import Event


class InteractionModel:
    def __init__(self, proximity_threshold: float = 80.0, min_interaction_frames: int = 15):
        self._proximity_threshold = proximity_threshold
        self._min_interaction_frames = min_interaction_frames
        self._owner_history: dict[int, int | None] = {}
        self._interaction_since: dict[tuple[int, int], int] = {}
        self._reported_interactions: set[tuple[int, int]] = set()
        self._follower_pairs: dict[int, int] = {}

    def update(self, tracks: list, frame: int) -> list[Event]:
        events = []
        persons = [t for t in tracks if t.class_name == "person"]
        others = [t for t in tracks if t.class_name != "person"]

        for obj in others:
            ocx = (obj.bbox[0] + obj.bbox[2]) // 2
            ocy = (obj.bbox[1] + obj.bbox[3]) // 2

            if not persons:
                self._owner_history[obj.id] = None
                continue

            nearest = min(
                persons,
                key=lambda p: np.linalg.norm(
                    np.array([ocx, ocy])
                    - np.array([(p.bbox[0] + p.bbox[2]) // 2, (p.bbox[1] + p.bbox[3]) // 2])
                ),
            )
            ncx = (nearest.bbox[0] + nearest.bbox[2]) // 2
            ncy = (nearest.bbox[1] + nearest.bbox[3]) // 2
            dist = np.linalg.norm(np.array([ocx, ocy]) - np.array([ncx, ncy]))

            if dist < self._proximity_threshold:
                self._owner_history[obj.id] = nearest.id
                pair = (obj.id, nearest.id)
                if pair not in self._interaction_since:
                    self._interaction_since[pair] = frame
                duration = frame - self._interaction_since[pair]
                if duration >= self._min_interaction_frames and pair not in self._reported_interactions:
                    self._reported_interactions.add(pair)
                    events.append(Event(
                        event_type="object_interaction",
                        track_id=obj.id,
                        class_name=obj.class_name,
                        duration=duration,
                        location=[ocx, ocy],
                        message=f"{obj.class_name} ID {obj.id} near person ID {nearest.id} for {duration}f",
                    ))
            else:
                self._interaction_since.pop((obj.id, nearest.id), None)

        self._update_follower_relationships(persons, frame, events)
        return events

    def _update_follower_relationships(self, persons: list, frame: int, events: list):
        if len(persons) < 2:
            return
        for i, p1 in enumerate(persons):
            p1cx = (p1.bbox[0] + p1.bbox[2]) // 2
            p1cy = (p1.bbox[1] + p1.bbox[3]) // 2
            for j, p2 in enumerate(persons):
                if i >= j:
                    continue
                p2cx = (p2.bbox[0] + p2.bbox[2]) // 2
                p2cy = (p2.bbox[1] + p2.bbox[3]) // 2
                dist = np.linalg.norm(np.array([p1cx, p1cy]) - np.array([p2cx, p2cy]))
                if dist < self._proximity_threshold * 1.5:
                    pair = (p1.id, p2.id)
                    if pair not in self._interaction_since:
                        self._interaction_since[pair] = frame
                    duration = frame - self._interaction_since[pair]
                    if duration >= self._min_interaction_frames * 4 and pair not in self._reported_interactions:
                        self._reported_interactions.add(pair)
                        events.append(Event(
                            event_type="group_traveling",
                            track_id=p1.id,
                            class_name="person",
                            duration=duration,
                            location=[(p1cx + p2cx) // 2, (p1cy + p2cy) // 2],
                            message=f"Persons {p1.id} and {p2.id} traveling together for {duration}f",
                        ))

    def get_owner(self, object_id: int) -> int | None:
        return self._owner_history.get(object_id)
