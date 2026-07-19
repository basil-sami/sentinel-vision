import time


class SpeedTracker:
    def __init__(self, window: int = 100):
        self._window = window
        self._timestamps: list[float] = []
        self._frame_count = 0
        self._start_time: float | None = None

    def start(self):
        self._start_time = time.time()
        self._timestamps = []
        self._frame_count = 0

    def tick(self):
        now = time.time()
        self._timestamps.append(now)
        self._frame_count += 1
        if len(self._timestamps) > self._window:
            self._timestamps.pop(0)

    @property
    def total_frames(self) -> int:
        return self._frame_count

    @property
    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def overall_fps(self) -> float:
        if self.elapsed > 0:
            return self._frame_count / self.elapsed
        return 0.0

    @property
    def instant_fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        window = self._timestamps[-1] - self._timestamps[0]
        if window > 0:
            return (len(self._timestamps) - 1) / window
        return 0.0

    def report(self) -> dict:
        return {
            "total_frames": self._frame_count,
            "elapsed_sec": round(self.elapsed, 2),
            "overall_fps": round(self.overall_fps, 1),
            "instant_fps": round(self.instant_fps, 1),
        }


def compute_mot_metrics(
    gt_tracks: dict[int, list[tuple[int, int, int, int, int]]],
    pred_tracks: dict[int, list[tuple[int, int, int, int, int]]],
    iou_threshold: float = 0.5,
) -> dict:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    id_switches = 0

    all_frames = set()
    for tids in gt_tracks.values():
        for f, *_ in tids:
            all_frames.add(f)
    for tids in pred_tracks.values():
        for f, *_ in tids:
            all_frames.add(f)

    prev_matches: dict[int, int] = {}

    for frame in sorted(all_frames):
        gt_bboxes = {}
        for tid, boxes in gt_tracks.items():
            matches = [(f, *b) for f, *b in boxes if f == frame]
            for m in matches:
                gt_bboxes[tid] = m[1:]

        pred_bboxes = {}
        for tid, boxes in pred_tracks.items():
            matches = [(f, *b) for f, *b in boxes if f == frame]
            for m in matches:
                pred_bboxes[tid] = m[1:]

        matched_gt = set()
        matched_pred = set()
        current_matches = {}

        for ptid, pbox in pred_bboxes.items():
            best_iou = iou_threshold
            best_gt = None
            for gtid, gbox in gt_bboxes.items():
                if gtid in matched_gt:
                    continue
                iou_val = _iou(pbox, gbox)
                if iou_val >= best_iou:
                    best_iou = iou_val
                    best_gt = gtid
            if best_gt is not None:
                matched_gt.add(best_gt)
                matched_pred.add(ptid)
                true_positives += 1
                current_matches[best_gt] = ptid
                if best_gt in prev_matches and prev_matches[best_gt] != ptid:
                    id_switches += 1
                prev_matches[best_gt] = ptid

        false_positives += len(pred_bboxes) - len(matched_pred)
        false_negatives += len(gt_bboxes) - len(matched_gt)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "mot_metrics": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "id_switches": id_switches,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        }
    }


def _iou(box1: tuple, box2: tuple) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0
