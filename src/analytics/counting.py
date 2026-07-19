class GateCounter:
    def __init__(self):
        self._entries: dict[str, int] = {}
        self._exits: dict[str, int] = {}
        self._counted: dict[str, set[int]] = {}

    def record(self, gate_name: str, track_id: int, direction: str):
        if track_id < 0:
            return
        counted = self._counted.setdefault(gate_name, set())
        event_key = (track_id, direction)
        if event_key in counted:
            return
        counted.add(event_key)
        if direction == "entering":
            self._entries[gate_name] = self._entries.get(gate_name, 0) + 1
        elif direction == "exiting":
            self._exits[gate_name] = self._exits.get(gate_name, 0) + 1

    def entries(self, gate_name: str) -> int:
        return self._entries.get(gate_name, 0)

    def exits(self, gate_name: str) -> int:
        return self._exits.get(gate_name, 0)

    def net(self, gate_name: str) -> int:
        return self.entries(gate_name) - self.exits(gate_name)

    def summary(self) -> dict:
        gates = set(self._entries.keys()) | set(self._exits.keys())
        return {
            g: {
                "entries": self.entries(g),
                "exits": self.exits(g),
                "net": self.net(g),
            }
            for g in sorted(gates)
        }
