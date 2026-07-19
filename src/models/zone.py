from dataclasses import dataclass, field

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    njit = lambda x: x
    _HAS_NUMBA = False


def _contains_nb(poly_flat: list[int], cx: int, cy: int) -> bool:
    n = len(poly_flat) // 2
    inside = False
    j = n - 1
    for i in range(n):
        xi = poly_flat[2 * i]
        yi = poly_flat[2 * i + 1]
        xj = poly_flat[2 * j]
        yj = poly_flat[2 * j + 1]
        if ((yi > cy) != (yj > cy)) and (
            cx < (xj - xi) * (cy - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


if _HAS_NUMBA:
    _contains_nb = njit(_contains_nb)


def _cross_direction_nb(
    x1: int, y1: int, x2: int, y2: int,
    px: int, py: int, cx: int, cy: int,
) -> int:
    s1 = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    s2 = (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)
    if s1 * s2 >= 0:
        return 0
    intersect_x = x1 + (x2 - x1) * ((y1 - py) / (cy - py)) if cy != py else cx
    if intersect_x < min(x1, x2) or intersect_x > max(x1, x2):
        return 0
    return 1 if s1 > 0 else -1


if _HAS_NUMBA:
    _cross_direction_nb = njit(_cross_direction_nb)


@dataclass
class Zone:
    name: str
    polygon: list[list[int]]
    zone_type: str = "generic"

    def contains(self, cx: int, cy: int) -> bool:
        if not self.polygon:
            return False
        poly_flat = [coord for pt in self.polygon for coord in pt]
        return _contains_nb(poly_flat, cx, cy)


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
        result = _cross_direction_nb(x1, y1, x2, y2, px, py, cx, cy)
        if result == 0:
            return None
        return "entering" if result == 1 else "exiting"
