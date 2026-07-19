import numpy as np


def filter_stationary(
    objects: list[dict],
    min_path_distance: float = 20.0,
    min_duration_frames: int = 5,
) -> list[dict]:
    if not objects:
        return []

    filtered = []
    for obj in objects:
        duration = obj.get("duration_frames", 0)
        path = obj.get("path", [])
        if len(path) < 2 or duration < min_duration_frames:
            continue

        total_distance = 0.0
        for i in range(1, len(path)):
            total_distance += np.linalg.norm(
                np.array(path[i]) - np.array(path[i - 1])
            )
        if total_distance >= min_path_distance:
            filtered.append(obj)

    return filtered
