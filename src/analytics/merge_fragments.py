import numpy as np


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def _iou(bbox_a, bbox_b):
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def merge_fragments(
    objects: list[dict],
    track_buffer: int = 300,
    max_center_distance: float = 100.0,
    min_iou: float = 0.1,
) -> list[dict]:
    if not objects:
        return []

    by_class: dict[str, list[dict]] = {}
    for obj in objects:
        by_class.setdefault(obj["class"], []).append(obj)

    merged_objects = {}
    next_id = 1

    for cls_name, cls_objects in by_class.items():
        cls_objects.sort(key=lambda o: o["path"][0]["frame"])

        for obj in cls_objects:
            obj_path = obj["path"]
            first_frame = obj_path[0]["frame"]
            last_frame = obj_path[-1]["frame"]
            first_bbox = obj_path[0]["bbox"]
            last_bbox = obj_path[-1]["bbox"]

            matched = False
            candidates = sorted(
                merged_objects.values(),
                key=lambda m: m["_last_frame"],
                reverse=True,
            )

            for existing in candidates:
                if existing["class"] != cls_name:
                    continue
                gap = first_frame - existing["_last_frame"]
                if gap <= 0 or gap > track_buffer:
                    continue

                existing_last_bbox = existing["_path"][-1]["bbox"]
                dist = np.linalg.norm(
                    _center(first_bbox) - _center(existing_last_bbox)
                )
                iou = _iou(first_bbox, existing_last_bbox)

                if dist > max_center_distance and iou < min_iou:
                    continue

                offset = len(existing["_path"])
                for entry in obj_path:
                    entry["id"] = existing["id"]
                    entry["path_index"] = offset
                    offset += 1
                existing["_path"].extend(obj_path)
                existing["_last_detection_frame"] = last_frame
                existing["_last_frame"] = last_frame
                matched = True
                break

            if not matched:
                new_id = next_id
                next_id += 1
                for entry in obj_path:
                    entry["id"] = new_id
                merged_objects[new_id] = {
                    "id": new_id,
                    "class": cls_name,
                    "class_id": obj["class_id"],
                    "path": list(obj_path),
                    "confidence": obj.get("confidence", 0),
                    "_path": list(obj_path),
                    "_last_frame": last_frame,
                }

    result = []
    for obj in merged_objects.values():
        path = obj["_path"]
        first_frame = path[0]["frame"]
        last_frame = path[-1]["frame"]
        result.append(
            {
                "id": obj["id"],
                "class": obj["class"],
                "class_id": obj["class_id"],
                "duration_frames": last_frame - first_frame + 1,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "path": path,
                "confidence": round(
                    np.mean([p.get("confidence", 0) for p in path]), 3
                ),
            }
        )

    result.sort(key=lambda o: (o["first_frame"], o["id"]))
    return result
