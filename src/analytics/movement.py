import numpy as np


_DIRECTIONS = ["north", "northeast", "east", "southeast",
               "south", "southwest", "west", "northwest"]


def estimate_direction(dx: float, dy: float) -> str:
    angle = np.degrees(np.arctan2(-dy, dx))
    if angle < 0:
        angle += 360
    index = round(angle / 45) % 8
    return _DIRECTIONS[index]


def movement_stats(path: list[list[int]]) -> dict:
    if len(path) < 2:
        return {
            "distance_pixels": 0,
            "average_speed": 0.0,
            "direction": "unknown",
        }

    total_distance = 0.0
    for i in range(1, len(path)):
        total_distance += np.linalg.norm(
            np.array(path[i]) - np.array(path[i - 1])
        )

    avg_speed = total_distance / len(path)
    dx = path[-1][0] - path[0][0]
    dy = path[-1][1] - path[0][1]
    direction = estimate_direction(dx, dy) if total_distance > 0 else "stationary"

    return {
        "distance_pixels": round(total_distance, 1),
        "average_speed": round(avg_speed, 2),
        "direction": direction,
    }
