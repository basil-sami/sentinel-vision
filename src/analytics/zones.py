from src.models.zone import Zone, LineGate


class ZoneManager:
    def __init__(self):
        self.zones: list[Zone] = []
        self.gates: list[LineGate] = []
        self._prev_positions: dict[int, tuple[int, int]] = {}

    def add_zone(self, zone: Zone):
        self.zones.append(zone)

    def add_gate(self, gate: LineGate):
        self.gates.append(gate)

    def get_zone(self, name: str) -> Zone | None:
        for z in self.zones:
            if z.name == name:
                return z
        return None

    def remove_zone(self, name: str):
        self.zones = [z for z in self.zones if z.name != name]

    def zones_at(self, cx: int, cy: int) -> list[Zone]:
        return [z for z in self.zones if z.contains(cx, cy)]

    def zones_for_track(self, track_id: int, cx: int, cy: int) -> list[Zone]:
        return self.zones_at(cx, cy)

    def check_gate_crossing(
        self, track_id: int, cx: int, cy: int
    ) -> list[tuple[str, str]]:
        results = []
        prev = self._prev_positions.get(track_id)
        self._prev_positions[track_id] = (cx, cy)
        if prev is None:
            return results
        for gate in self.gates:
            direction = gate.cross_direction(prev, (cx, cy))
            if direction:
                results.append((gate.name, direction))
        return results

    def get_config(self) -> dict:
        return {
            "zones": [
                {"name": z.name, "type": z.zone_type, "polygon": z.polygon}
                for z in self.zones
            ],
            "gates": [
                {"name": g.name, "p1": g.p1, "p2": g.p2}
                for g in self.gates
            ],
        }

    @classmethod
    def from_config(cls, config: dict) -> "ZoneManager":
        mgr = cls()
        for zc in config.get("zones", []):
            mgr.add_zone(Zone(
                name=zc["name"],
                polygon=zc["polygon"],
                zone_type=zc.get("type", "generic"),
            ))
        for gc in config.get("gates", []):
            mgr.add_gate(LineGate(
                name=gc["name"],
                p1=gc["p1"],
                p2=gc["p2"],
            ))
        return mgr
