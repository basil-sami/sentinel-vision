"""Persistent store of global identities with their ReID embeddings.

JSON file with base64-encoded embeddings (same pattern as FaceGallery).
Thread-safe for multi-camera ingestion via a lock.
"""

import base64
import json
import threading
import time
from pathlib import Path

import numpy as np

from src.identity.global_identity import GlobalIdentity, Observation

EMBED_CAP = 8  # max representative embeddings kept per identity


class IdentityStore:
    def __init__(self, path: str = "identity_store.json"):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._identities: dict[int, GlobalIdentity] = {}
        self._next_id = 1
        self._load()

    def _load(self):
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
            for ident in raw.get("identities", []):
                gid = GlobalIdentity(
                    id=ident["id"],
                    class_name=ident["class_name"],
                    first_seen=ident.get("first_seen", 0.0),
                    last_seen=ident.get("last_seen", 0.0),
                )
                for emb_b64 in ident.get("embeddings_b64", []):
                    gid.embeddings.append(
                        np.frombuffer(base64.b64decode(emb_b64), dtype=np.float32).copy()
                    )
                for obs in ident.get("observations", []):
                    gid.observations.append(Observation(
                        camera_id=obs["camera_id"],
                        local_track_id=obs["local_track_id"],
                        class_name=obs.get("class_name", ""),
                        first_frame=obs["first_frame"],
                        last_frame=obs["last_frame"],
                        duration_frames=obs.get("duration_frames", 0),
                        matched_at=obs.get("matched_at", 0.0),
                        match_score=obs.get("match_score", 0.0),
                    ))
                self._identities[gid.id] = gid
                self._next_id = max(self._next_id, gid.id + 1)
        except Exception:
            self._identities = {}
            self._next_id = 1

    def save(self):
        with self._lock:
            raw = {
                "next_id": self._next_id,
                "identities": [
                    {
                        "id": g.id,
                        "class_name": g.class_name,
                        "first_seen": g.first_seen,
                        "last_seen": g.last_seen,
                        "embeddings_b64": [
                            base64.b64encode(e.tobytes()).decode()
                            for e in g.embeddings
                        ],
                        "observations": [
                            {
                                "camera_id": o.camera_id,
                                "local_track_id": o.local_track_id,
                                "class_name": o.class_name,
                                "first_frame": o.first_frame,
                                "last_frame": o.last_frame,
                                "duration_frames": o.duration_frames,
                                "matched_at": o.matched_at,
                                "match_score": o.match_score,
                            }
                            for o in g.observations
                        ],
                    }
                    for g in self._identities.values()
                ],
            }
            self._path.write_text(json.dumps(raw, indent=2))

    # ── reads ──

    def get(self, gid: int) -> GlobalIdentity | None:
        return self._identities.get(gid)

    def all(self) -> list[GlobalIdentity]:
        return list(self._identities.values())

    def count(self) -> int:
        return len(self._identities)

    def embeddings(self, class_name: str | None = None) -> list[tuple[int, np.ndarray]]:
        """All (global_id, embedding) pairs, optionally filtered by class."""
        out = []
        for gid, ident in self._identities.items():
            if class_name and ident.class_name != class_name:
                continue
            for emb in ident.embeddings:
                out.append((gid, emb))
        return out

    # ── writes ──

    def create(self, class_name: str, embedding: np.ndarray,
               obs: Observation) -> int:
        with self._lock:
            gid = self._next_id
            self._next_id += 1
            ident = GlobalIdentity(id=gid, class_name=class_name)
            self._add_embedding(ident, embedding)
            ident.add_observation(obs)
            self._identities[gid] = ident
            return gid

    def add_embedding_and_observation(self, gid: int, embedding: np.ndarray,
                                      obs: Observation):
        with self._lock:
            ident = self._identities.get(gid)
            if ident is None:
                return
            self._add_embedding(ident, embedding)
            ident.add_observation(obs)

    @staticmethod
    def _add_embedding(ident: GlobalIdentity, embedding: np.ndarray):
        norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        ident.embeddings.append(norm)
        if len(ident.embeddings) > EMBED_CAP:
            ident.embeddings.pop(0)  # drop oldest viewpoint
