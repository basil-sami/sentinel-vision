import numpy as np


def merge_fragments(
    objects: list[dict],
    track_buffer: int = 300,
    max_center_distance: float = 100.0,
) -> list[dict]:
    if not objects:
        return []

    by_class: dict[str, list[dict]] = {}
    for obj in objects:
        by_class.setdefault(obj["class"], []).append(obj)

    merged = {}
    next_id = 1

    for cls_name, cls_objects in by_class.items():
        cls_objects.sort(key=lambda o: o["first_frame"])

        for obj in cls_objects:
            first_frame = obj["first_frame"]
            last_frame = obj["last_frame"]
            path = obj["path"]
            first_pt = path[0] if path else [0, 0]
            last_pt = path[-1] if path else [0, 0]

            matched = False
            candidates = sorted(
                merged.values(),
                key=lambda m: m["_last_frame"],
                reverse=True,
            )

            for existing in candidates:
                if existing["class"] != cls_name:
                    continue
                gap = first_frame - existing["_last_frame"]
                if gap <= 0 or gap > track_buffer:
                    continue

                existing_last_pt = existing["_path"][-1]
                dist = np.linalg.norm(
                    np.array(first_pt) - np.array(existing_last_pt)
                )

                if dist > max_center_distance:
                    continue

                existing["_path"].extend(path)
                existing["_last_frame"] = last_frame
                matched = True
                break

            if not matched:
                merged[next_id] = {
                    "id": next_id,
                    "class": cls_name,
                    "class_id": obj["class_id"],
                    "_first_frame": first_frame,
                    "_path": list(path),
                    "_last_frame": last_frame,
                }
                next_id += 1

    result = []
    for obj in merged.values():
        path = obj["_path"]
        result.append(
            {
                "id": obj["id"],
                "class": obj["class"],
                "class_id": obj["class_id"],
                "duration_frames": obj["_last_frame"] - obj["_first_frame"] + 1,
                "first_frame": obj["_first_frame"],
                "last_frame": obj["_last_frame"],
                "path": path,
                "confidence": 0.0,
            }
        )

    result.sort(key=lambda o: (o["first_frame"], o["id"]))
    return result
