from dataclasses import dataclass, field


@dataclass
class Zone:
    name: str
    polygon: list[list[int]]
    zone_type: str = "generic"

    def contains(self, cx: int, cy: int) -> bool:
        if not self.polygon:
            return False
        n = len(self.polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = self.polygon[i]
            xj, yj = self.polygon[j]
            if ((yi > cy) != (yj > cy)) and (
                cx < (xj - xi) * (cy - yi) / (yj - yi) + xi
            ):
                inside = not inside
            j = i
        return inside


@dataclass
class LineGate:
    name: str
    p1: list[int]
    p2: list[int]

    def cross_direction(self, prev: tuple[int, int], curr: tuple[int, int]) -> str | None:
        px, py = prev
        cx, cy = curr
        x1, y1 = self.p1
        x2, y2 = self.p2

        def side(px, py):
            return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

        s1 = side(px, py)
        s2 = side(cx, cy)

        if s1 * s2 >= 0:
            return None

        intersect_x = x1 + (x2 - x1) * ((y1 - py) / (cy - py)) if cy != py else cx
        if intersect_x < min(x1, x2) or intersect_x > max(x1, x2):
            return None

        return "entering" if s1 > 0 else "exiting"
