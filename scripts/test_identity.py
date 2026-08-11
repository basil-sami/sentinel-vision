"""Verify the global identity layer end-to-end with synthetic embeddings.

Runs without ML deps (numpy only):
  1. Topology parser against the real configs/4cameras.json
  2. Cross-camera matching — same person re-observed on a neighbor camera
     must get the same global_id; a different person must NOT be conflated
  3. Per-identity timeline output

Run:  python3 scripts/test_identity.py
"""

import base64
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

from src.identity.identity_manager import IdentityManager  # noqa: E402
from src.identity.matcher import cosine_similarity  # noqa: E402


def unit(rng, seed):
    e = rng.normal(size=512).astype(np.float32)
    return e / np.linalg.norm(e)


def same_person(base, sigma=0.08):
    rng = np.random.default_rng(123)
    noise = rng.normal(size=512).astype(np.float32)
    noise = noise / np.linalg.norm(noise)
    e = base + sigma * noise
    return e / np.linalg.norm(e)


def b64(e):
    return base64.b64encode(e.tobytes()).decode()


def main() -> int:
    cfg = json.loads((SRC_DIR / "configs" / "4cameras.json").read_text())
    topo = IdentityManager.load_topology(cfg)
    assert topo.get("cam00_original") == {"cam01_mirror_slow"}, topo
    assert topo.get("cam03_slow_reverse") == {"cam02_mirror_reverse"}, topo
    print("1. topology OK (by camera name):", len(topo), "cameras")

    rng = np.random.default_rng(42)
    p1 = unit(rng, 1)
    p2 = unit(rng, 2)
    car = unit(rng, 3)
    p1_again = same_person(p1)
    assert cosine_similarity(p1, p1_again) > 0.95
    assert cosine_similarity(p1, p2) < 0.5
    print("2. embedding similarity OK")

    mgr = IdentityManager(
        store_path=os.path.join(tempfile.mkdtemp(), "id.json"), topology=topo
    )
    data = {
        "cam00_original": [
            {"id": 1, "class": "person", "first_frame": 0, "last_frame": 120,
             "duration_frames": 121, "camera_id": "cam00_original",
             "embedding_b64": b64(p1)},
            {"id": 2, "class": "vehicle", "first_frame": 5, "last_frame": 90,
             "duration_frames": 86, "camera_id": "cam00_original",
             "embedding_b64": b64(car)},
        ],
        "cam01_mirror_slow": [
            {"id": 9, "class": "person", "first_frame": 300, "last_frame": 420,
             "duration_frames": 121, "camera_id": "cam01_mirror_slow",
             "embedding_b64": b64(p1_again)},
        ],
        "cam02_mirror_reverse": [
            {"id": 4, "class": "person", "first_frame": 500, "last_frame": 620,
             "duration_frames": 121, "camera_id": "cam02_mirror_reverse",
             "embedding_b64": b64(p2)},
        ],
    }
    summary = mgr.ingest({**data})
    g00 = data["cam00_original"][0]["global_id"]
    g01 = data["cam01_mirror_slow"][0]["global_id"]
    g02 = data["cam02_mirror_reverse"][0]["global_id"]

    assert g00 == g01, f"same person failed to match ({g00} vs {g01})"
    assert g00 != g02 and g01 != g02, "different person wrongly conflated"
    assert summary["multi_camera_identities"] == 1
    print("3. cross-camera matching OK (person seen on 2+ cameras =",
          summary["multi_camera_identities"], ")")

    tl = mgr.timelines()[g00]
    assert len(tl) == 2 and tl[1]["camera_id"] == "cam01_mirror_slow"
    print("4. timeline OK:", [(t["camera_id"], t["first_frame"]) for t in tl])

    print("\nALL IDENTITY TESTS PASSED")


if __name__ == "__main__":
    main()
