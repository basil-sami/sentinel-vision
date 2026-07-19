from enum import Enum


class TrackState(Enum):
    NEW = "new"
    ACTIVE = "active"
    OCCLUDED = "occluded"
    LOST = "lost"
    MERGED = "merged"
    ENDED = "ended"


_TRANSITIONS: dict[TrackState, set[TrackState]] = {
    TrackState.NEW: {TrackState.ACTIVE, TrackState.LOST, TrackState.ENDED},
    TrackState.ACTIVE: {TrackState.ACTIVE, TrackState.OCCLUDED, TrackState.ENDED},
    TrackState.OCCLUDED: {TrackState.ACTIVE, TrackState.LOST, TrackState.ENDED},
    TrackState.LOST: {TrackState.ACTIVE, TrackState.ENDED, TrackState.MERGED},
    TrackState.MERGED: {TrackState.ENDED},
    TrackState.ENDED: set(),
}


class TrackStateMachine:
    def __init__(self):
        self._states: dict[int, TrackState] = {}
        self._history: dict[int, list[tuple[int, TrackState]]] = {}

    def init_track(self, track_id: int, frame: int):
        self._states[track_id] = TrackState.NEW
        self._history[track_id] = [(frame, TrackState.NEW)]

    def transition(self, track_id: int, new_state: TrackState, frame: int) -> bool:
        current = self._states.get(track_id)
        if current is None:
            self.init_track(track_id, frame)
            current = TrackState.NEW

        if new_state == current:
            return True

        if new_state in _TRANSITIONS.get(current, set()):
            self._states[track_id] = new_state
            self._history[track_id].append((frame, new_state))
            return True
        return False

    def get_state(self, track_id: int) -> TrackState:
        return self._states.get(track_id, TrackState.NEW)

    def get_state_name(self, track_id: int) -> str:
        return self.get_state(track_id).value

    def is_active(self, track_id: int) -> bool:
        s = self.get_state(track_id)
        return s in (TrackState.NEW, TrackState.ACTIVE, TrackState.OCCLUDED)

    def history(self, track_id: int) -> list[tuple[int, TrackState]]:
        return self._history.get(track_id, [])

    def summary(self) -> dict:
        counts = {}
        for s in TrackState:
            counts[s.value] = 0
        for s in self._states.values():
            counts[s.value] = counts.get(s.value, 0) + 1
        return counts
