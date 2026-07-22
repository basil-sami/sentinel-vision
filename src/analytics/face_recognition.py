"""Face detection and recognition for person tracks.

Uses insightface (RetinaFace + ArcFace) when available.
Gallery persisted as JSON with base64-encoded embeddings.

Heavy GPU inference (model.get) is offloaded to a thread pool so the
main pipeline never blocks on face recognition.
"""

import base64
import json
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------

class FaceGallery:
    """Persistent store of known faces with their embeddings."""

    def __init__(self, path: str = "face_gallery.json"):
        self._path = Path(path)
        self._entries: dict[str, dict] = {}  # name → {"embedding": np.ndarray, "added": float, "count": int}
        self._load()

    def _load(self):
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
            for name, data in raw.items():
                emb_bytes = base64.b64decode(data["embedding"])
                self._entries[name] = {
                    "embedding": np.frombuffer(emb_bytes, dtype=np.float32).copy(),
                    "added": data.get("added", 0),
                    "count": data.get("count", 1),
                }
            log.info("Loaded %d known faces from %s", len(self._entries), self._path)
        except Exception as e:
            log.warning("Failed to load face gallery from %s: %s", self._path, e)

    def save(self):
        raw = {}
        for name, data in self._entries.items():
            raw[name] = {
                "embedding": base64.b64encode(data["embedding"].tobytes()).decode(),
                "added": data.get("added", 0),
                "count": data.get("count", 1),
            }
        self._path.write_text(json.dumps(raw, indent=2))
        log.info("Saved %d known faces to %s", len(self._entries), self._path)

    def add(self, name: str, embedding: np.ndarray):
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        if name in self._entries:
            # Average with existing embedding
            old = self._entries[name]["embedding"]
            count = self._entries[name]["count"]
            self._entries[name]["embedding"] = (old * count + emb_norm) / (count + 1)
            self._entries[name]["count"] = count + 1
        else:
            self._entries[name] = {
                "embedding": emb_norm,
                "added": time.time(),
                "count": 1,
            }
        self.save()

    def match(self, embedding: np.ndarray, threshold: float = 0.45) -> tuple[str | None, float]:
        """Return (name, similarity) for best match above threshold."""
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        best_name = None
        best_sim = 0.0
        for name, data in self._entries.items():
            sim = float(np.dot(emb_norm, data["embedding"]))
            sim = max(-1.0, min(1.0, sim))  # clamp
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_sim >= threshold:
            return best_name, best_sim
        return None, best_sim

    def known_names(self) -> list[str]:
        return list(self._entries.keys())

    def remove(self, name: str):
        self._entries.pop(name, None)
        self.save()


# ---------------------------------------------------------------------------
# Face Recognizer
# ---------------------------------------------------------------------------

class FaceRecognizer:
    """Detects faces in person crops, extracts embeddings, matches gallery.

    Uses insightface (ArcFace) on GPU for high-throughput recognition.
    Falls back to a no-op if insightface is not installed.
    """

    def __init__(self, gallery_path: str = "face_gallery.json", device: str = "cuda",
                 min_face_size: int = 40, match_threshold: float = 0.45,
                 confirm_frames: int = 5,
                 capture_unknowns: bool = True,
                 unknown_dir: str = "unknown_faces",
                 max_workers: int = 1):
        self._gallery = FaceGallery(gallery_path)
        self._device = device
        self._min_face_size = min_face_size
        self._match_threshold = match_threshold
        self._confirm_frames = confirm_frames
        self._capture_unknowns = capture_unknowns

        # Track identity state
        self._track_identity: dict[int, str] = {}          # track_id → name
        self._track_confidence: dict[int, float] = {}      # track_id → avg similarity
        self._track_embeddings: dict[int, list[np.ndarray]] = {}  # track_id → [embeddings]
        self._track_name_counts: dict[int, dict[str, int]] = {}   # track_id → {name: count}

        # Unknown face store
        self._unknown_store = UnknownFaceStore(unknown_dir) if capture_unknowns else None

        # Async inference offload
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending_faces: dict[int, tuple[Future, int]] = {}  # track_id → (future, frame_idx)

        # Face interval throttling (process each track at most every N frames)
        self._face_interval = 6  # ~4 FPS at 25 FPS
        self._last_process_frame: dict[int, int] = {}

        # Load model
        self._model = self._load_model()

    def _load_model(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider" if self._device == "cuda" else "CPUExecutionProvider"])
            app.prepare(ctx_id=0 if self._device == "cuda" else -1, det_size=(320, 320))
            log.info("FaceRecognizer: loaded insightface buffalo_l")
            return app
        except ImportError:
            log.warning("FaceRecognizer: insightface not installed. Run: pip install insightface")
            return None
        except Exception as e:
            log.warning("FaceRecognizer: failed to load model: %s", e)
            return None

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def gallery(self) -> FaceGallery:
        return self._gallery

    def process_frame(self, frame: np.ndarray, tracks: list,
                      frame_idx: int) -> list[dict]:
        """Process all person tracks for face recognition — async.

        Heavy GPU inference (insightface model.get) runs in a thread pool
        so the main pipeline is never blocked. Results are collected on
        subsequent frames with a 1-2 frame delay, which is fine since
        identity confirmation requires multiple frames anyway.

        Returns list of events from *completed* async results.
        """
        events = []
        if not self.available:
            return events

        # 1. Collect completed async results
        self._collect_pending(events, frame_idx)

        # 2. Submit new face crops for tracks not yet identified
        for t in tracks:
            if t.class_name != "person":
                continue
            if t.id in self._track_identity:
                continue  # already confirmed
            if t.id in self._pending_faces:
                continue  # already pending from a previous frame

            # Throttle: skip if we processed this track recently
            last = self._last_process_frame.get(t.id, -1)
            if frame_idx - last < self._face_interval:
                continue

            face_roi = self._crop_face_region(frame, t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3])
            if face_roi is None:
                continue

            self._last_process_frame[t.id] = frame_idx
            future = self._executor.submit(self._run_face_inference,
                                           face_roi, t.id, frame_idx)
            self._pending_faces[t.id] = (future, frame_idx)

        return events

    def _run_face_inference(self, face_roi: np.ndarray,
                            track_id: int, frame_idx: int) -> dict | None:
        """Run GPU face inference in a background thread."""
        try:
            faces = self._model.get(face_roi)
        except Exception:
            return None
        if not faces:
            return None

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embedding = face.normed_embedding
        if embedding is None:
            return None

        name, sim = self._gallery.match(embedding, self._match_threshold)
        return {
            "track_id": track_id,
            "frame": frame_idx,
            "embedding": embedding,
            "name": name,
            "sim": sim,
            "det_score": float(face.det_score) if hasattr(face, 'det_score') else 0.0,
            "face_roi": face_roi,
        }

    def _collect_pending(self, events: list, current_frame: int):
        """Collect completed async face inference results."""
        done_ids = []
        for tid, (future, submit_frame) in self._pending_faces.items():
            if not future.done():
                continue
            done_ids.append(tid)

            try:
                result = future.result()
            except Exception:
                continue
            if result is None:
                continue

            tid_ = result["track_id"]
            embedding = result["embedding"]
            name = result["name"]
            sim = result["sim"]

            if name is None:
                # Unknown face
                if self._capture_unknowns and self._unknown_store is not None:
                    self._unknown_store.capture(
                        tid_, submit_frame, result["face_roi"], embedding,
                        result["det_score"],
                    )
                continue

            # Accumulate votes across frames
            self._track_name_counts.setdefault(tid_, {})[name] = \
                self._track_name_counts[tid_].get(name, 0) + 1
            self._track_embeddings.setdefault(tid_, []).append(embedding)

            total = sum(self._track_name_counts[tid_].values())
            best_name = max(self._track_name_counts[tid_],
                            key=self._track_name_counts[tid_].get)
            best_count = self._track_name_counts[tid_][best_name]

            if best_count >= self._confirm_frames and best_count / total >= 0.6:
                all_embs = self._track_embeddings[tid_]
                avg_emb = np.mean(all_embs, axis=0)
                avg_emb /= np.linalg.norm(avg_emb) + 1e-8

                self._track_identity[tid_] = best_name
                self._track_confidence[tid_] = float(
                    np.dot(avg_emb, self._gallery._entries[best_name]["embedding"]))
                events.append({
                    "type": "face_recognized",
                    "track_id": tid_,
                    "name": best_name,
                    "confidence": round(self._track_confidence[tid_], 3),
                    "frame": submit_frame,
                })
                log.info("Recognized track %d as '%s' (sim=%.3f)",
                         tid_, best_name, self._track_confidence[tid_])

        for tid in done_ids:
            self._pending_faces.pop(tid, None)

    def _crop_face_region(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray | None:
        """Extract upper portion of person bbox where face is expected."""
        h = y2 - y1
        w = x2 - x1
        face_y1 = max(0, y1)
        face_y2 = max(y1 + 1, min(frame.shape[0], y1 + int(h * 0.35)))
        face_x1 = max(0, x1)
        face_x2 = min(frame.shape[1], x2)
        if face_y2 - face_y1 < self._min_face_size or face_x2 - face_x1 < self._min_face_size:
            return None
        return frame[face_y1:face_y2, face_x1:face_x2]

    def get_track_identity(self, track_id: int) -> tuple[str | None, float]:
        name = self._track_identity.get(track_id)
        conf = self._track_confidence.get(track_id, 0.0)
        return name, conf

    def get_all_identities(self) -> dict[int, tuple[str, float]]:
        return {tid: (self._track_identity[tid], self._track_confidence[tid])
                for tid in self._track_identity}

    def add_known_face(self, name: str, image: np.ndarray) -> bool:
        """Register a new face from an image."""
        if not self.available:
            return False
        faces = self._model.get(image)
        if not faces:
            log.warning("No face detected in provided image for '%s'", name)
            return False
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        if face.normed_embedding is None:
            return False
        self._gallery.add(name, face.normed_embedding)
        log.info("Added known face: '%s'", name)
        return True

    def add_known_face_from_file(self, name: str, image_path: str) -> bool:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            log.warning("Cannot read image: %s", image_path)
            return False
        return self.add_known_face(name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    @property
    def unknown_store(self) -> "UnknownFaceStore":
        return self._unknown_store


# ---------------------------------------------------------------------------
# Unknown Face Capture
# ---------------------------------------------------------------------------

class UnknownFaceStore:
    """Persistent store of unknown (unrecognized) face captures.

    Each unknown face is saved as:
      unknown_faces/unknown_{seq:04d}.jpg  (face crop image)
      unknown_faces/index.json             (embedding + metadata)

    A separate script can batch-match these against a known-face database
    and promote them to the gallery. Only the first sighting of each track
    is captured (deduplicated per run via _captured_tracks).
    """

    def __init__(self, storage_dir: str = "unknown_faces"):
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.json"
        self._index: list[dict] = []
        self._seq = 0
        self._captured_tracks: set[int] = set()  # per-run, avoid dupes
        self._load()

    def _load(self):
        if self._index_path.exists():
            try:
                self._index = json.loads(self._index_path.read_text())
                self._seq = len(self._index)
            except Exception:
                self._index = []

    def save(self):
        self._index_path.write_text(json.dumps(self._index, indent=2))

    def capture(self, track_id: int, frame_idx: int,
                face_crop: np.ndarray, embedding: np.ndarray,
                det_score: float) -> str:
        """Save an unknown face and return its ID."""
        if track_id in self._captured_tracks:
            return ""
        self._captured_tracks.add(track_id)

        uid = f"unknown_{self._seq:04d}"
        img_path = self._dir / f"{uid}.jpg"
        import cv2
        cv2.imwrite(str(img_path), cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))

        entry = {
            "id": uid,
            "track_id": track_id,
            "frame": frame_idx,
            "image": str(img_path.relative_to(self._dir.parent) if img_path.parent != self._dir.parent else str(img_path)),
            "det_score": round(det_score, 3),
            "embedding": base64.b64encode(embedding.astype(np.float32).tobytes()).decode(),
            "captured_at": time.time(),
            "matched_name": None,
            "match_confidence": None,
        }
        self._index.append(entry)
        self._seq += 1
        self.save()
        log.info("Captured unknown face %s (track %d, frame %d)", uid, track_id, frame_idx)
        return uid

    def list_unknowns(self) -> list[dict]:
        """Return all unknown entries with embedding decoded."""
        result = []
        for entry in self._index:
            e = dict(entry)
            emb_bytes = base64.b64decode(e["embedding"])
            e["embedding"] = np.frombuffer(emb_bytes, dtype=np.float32).copy()
            result.append(e)
        return result

    def list_unmatched(self) -> list[dict]:
        """Return only unknown entries that haven't been matched yet."""
        return [e for e in self.list_unknowns() if e["matched_name"] is None]

    def mark_matched(self, uid: str, name: str, confidence: float):
        for entry in self._index:
            if entry["id"] == uid:
                entry["matched_name"] = name
                entry["match_confidence"] = round(confidence, 3)
                break
        self.save()

    def promote_to_gallery(self, uid: str, name: str,
                           face_recognizer: "FaceRecognizer") -> bool:
        """Promote an unknown face to the known gallery with a name."""
        for entry in self._index:
            if entry["id"] != uid:
                continue
            if entry["matched_name"] and entry["matched_name"] != name:
                log.warning("Overwriting previous match for %s", uid)
            emb_bytes = base64.b64decode(entry["embedding"])
            embedding = np.frombuffer(emb_bytes, dtype=np.float32).copy()
            face_recognizer.gallery.add(name, embedding)
            entry["matched_name"] = name
            entry["match_confidence"] = 1.0
            self.save()
            log.info("Promoted %s → '%s'", uid, name)
            return True
        return False

    def count(self) -> int:
        return len(self._index)
