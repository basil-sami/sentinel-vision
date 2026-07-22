"""Face detection and recognition for person tracks.

Uses insightface (RetinaFace + ArcFace) when available.

Gallery persisted as JSON with base64-encoded embeddings.
"""

import base64
import json
import logging
import time
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
                         confirm_frames: int = 5):
        self._gallery = FaceGallery(gallery_path)
        self._device = device
        self._min_face_size = min_face_size
        self._match_threshold = match_threshold
        self._confirm_frames = confirm_frames

        # Track identity state
        self._track_identity: dict[int, str] = {}          # track_id → name
        self._track_confidence: dict[int, float] = {}      # track_id → avg similarity
        self._track_embeddings: dict[int, list[np.ndarray]] = {}  # track_id → [embeddings]
        self._track_name_counts: dict[int, dict[str, int]] = {}   # track_id → {name: count}

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
        """Process all person tracks in a frame for face recognition.

        Returns list of events: [{"type": "face_recognized", "track_id": ..., "name": ..., "confidence": ...}]
        """
        events = []
        if not self.available:
            return events

        for t in tracks:
            if t.class_name != "person":
                continue
            if t.id in self._track_identity:
                continue  # already confirmed

            x1, y1, x2, y2 = t.bbox
            face_roi = self._crop_face_region(frame, x1, y1, x2, y2)
            if face_roi is None:
                continue

            try:
                faces = self._model.get(face_roi)
            except Exception:
                continue

            if not faces:
                continue

            # Take the largest face in the crop
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embedding = face.normed_embedding
            if embedding is None:
                continue

            name, sim = self._gallery.match(embedding, self._match_threshold)
            if name is None:
                continue

            # Accumulate votes across frames
            self._track_name_counts.setdefault(t.id, {})[name] = self._track_name_counts[t.id].get(name, 0) + 1
            self._track_embeddings.setdefault(t.id, []).append(embedding)

            total = sum(self._track_name_counts[t.id].values())
            best_name = max(self._track_name_counts[t.id], key=self._track_name_counts[t.id].get)
            best_count = self._track_name_counts[t.id][best_name]

            if best_count >= self._confirm_frames and best_count / total >= 0.6:
                # Confirm identity
                all_embs = self._track_embeddings[t.id]
                avg_emb = np.mean(all_embs, axis=0)
                avg_emb /= np.linalg.norm(avg_emb) + 1e-8

                self._track_identity[t.id] = best_name
                self._track_confidence[t.id] = float(np.dot(avg_emb, self._gallery._entries[best_name]["embedding"]))
                events.append({
                    "type": "face_recognized",
                    "track_id": t.id,
                    "name": best_name,
                    "confidence": round(self._track_confidence[t.id], 3),
                    "frame": frame_idx,
                })
                log.info("Recognized track %d as '%s' (sim=%.3f)", t.id, best_name, self._track_confidence[t.id])

        return events

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
