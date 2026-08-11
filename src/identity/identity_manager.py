"""IdentityManager — cross-camera identity assignment orchestrator.

Consumes per-camera pipeline outputs (objects with `embedding_b64` +
`camera_id`), matches against the persistent IdentityStore, assigns
`global_id`s, and emits a global identity report with per-identity
multi-camera timelines.

Usage (after all cameras complete):
    mgr = IdentityManager(store_path, topology=...)
    report = mgr.ingest({cam_id: result["objects"]})
"""

import base64
import json
import time
from pathlib import Path

import numpy as np

from src.identity.embedding_store import IdentityStore
from src.identity.global_identity import Observation
from src.identity.matcher import find_best_match


class IdentityManager:
    def __init__(self, store_path: str = "identity_store.json",
                 topology: dict[str, set[str]] | None = None):
        self.store = IdentityStore(store_path)
        self.topology = topology or {}

    @staticmethod
    def _decode_embedding(emb_b64: str | None) -> np.ndarray | None:
        if not emb_b64:
            return None
        try:
            return np.frombuffer(base64.b64decode(emb_b64), dtype=np.float32).copy()
        except Exception:
            return None

    def _identity_cameras(self) -> dict[int, set[str]]:
        return {g.id: set(g.cameras) for g in self.store.all()}

    def ingest(self, camera_objects: dict[str, list[dict]]) -> dict:
        """Assign global_ids across cameras. Mutates objects in place.

        camera_objects: {camera_id: [object dicts from result["objects"]]}
        Returns summary dict.
        """
        now = time.time()
        matched = 0
        new_created = 0
        uncertain = 0
        skipped = 0

        for cam_id, objects in camera_objects.items():
            # Recompute candidate cache per camera so earlier cameras'
            # assignments influence later ones (chronological ingest)
            candidates = self.store.embeddings()
            identity_cams = self._identity_cameras()

            for obj in objects:
                emb = self._decode_embedding(obj.get("embedding_b64"))
                if emb is None:
                    skipped += 1
                    continue

                obs = Observation(
                    camera_id=cam_id,
                    local_track_id=obj.get("id", -1),
                    class_name=obj.get("class", ""),
                    first_frame=obj.get("first_frame", 0),
                    last_frame=obj.get("last_frame", 0),
                    duration_frames=obj.get("duration_frames", 0),
                    matched_at=now,
                )

                gid, score, confident = find_best_match(
                    emb, candidates,
                    identity_cameras=identity_cams,
                    camera_id=cam_id,
                    topology=self.topology,
                )

                if gid is None:
                    gid = self.store.create(obj.get("class", "unknown"), emb, obs)
                    new_created += 1
                else:
                    obs.match_score = score
                    self.store.add_embedding_and_observation(gid, emb, obs)
                    if confident:
                        matched += 1
                    else:
                        uncertain += 1

                obj["global_id"] = gid
                obj["identity_match_score"] = round(score, 3)

        self.store.save()

        return {
            "total_objects": sum(len(v) for v in camera_objects.values()),
            "matched": matched,
            "uncertain_matches": uncertain,
            "new_identities": new_created,
            "skipped_no_embedding": skipped,
            "total_global_identities": self.store.count(),
            "multi_camera_identities": sum(
                1 for g in self.store.all() if g.camera_count > 1
            ),
        }

    def timelines(self) -> dict[int, list[dict]]:
        """Per-global-id cross-camera timelines: [{camera, frames, track}]."""
        out = {}
        for g in self.store.all():
            out[g.id] = [
                {
                    "camera_id": o.camera_id,
                    "local_track_id": o.local_track_id,
                    "first_frame": o.first_frame,
                    "last_frame": o.last_frame,
                    "duration_frames": o.duration_frames,
                    "match_score": round(o.match_score, 3),
                }
                for o in sorted(g.observations, key=lambda o: o.first_frame)
            ]
        return out

    def write_report(self, output_path: str | Path, summary: dict,
                     extra: dict | None = None) -> Path:
        path = Path(output_path)
        report = {
            "generated_at": time.time(),
            "summary": summary,
            "identities": [
                g.to_dict() for g in sorted(self.store.all(), key=lambda g: g.id)
            ],
            "timelines": self.timelines(),
        }
        if extra:
            report.update(extra)
        path.write_text(json.dumps(report, indent=2))
        return path

    @staticmethod
    def load_topology(config: dict | None) -> dict[str, set[str]]:
        """Build camera→neighbors map from a topology config.

        Accepts {"cameras": [{"id": ..., "topology": {"adjacent": [...]}}]}
        or {camera_id: [adjacent_ids]}.
        """
        if not config:
            return {}
        topo: dict[str, set[str]] = {}
        if "cameras" in config:
            for cam in config["cameras"]:
                cid = cam.get("id") or cam.get("name", "")
                adj = []
                if isinstance(cam.get("topology"), dict):
                    adj = cam["topology"].get("adjacent", [])
                elif isinstance(cam.get("topology"), list):
                    adj = cam["topology"]
                if cid and adj:
                    topo[cid] = set(adj)
        else:
            for cid, adj in config.items():
                if isinstance(adj, (list, set)):
                    topo[str(cid)] = set(adj)
        return topo
