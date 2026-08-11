"""Cross-camera identity matching.

Cosine similarity between ReID embeddings with threshold gating:
    >= CONFIRMED_THRESHOLD (0.85)  → same identity, confident
    >= UNCERTAIN_THRESHOLD (0.70)  → same identity, low confidence
    <  UNCERTAIN_THRESHOLD         → new identity

Topology boost: if the candidate identity was already seen on a camera
adjacent to the current observation's camera, its score is nudged up.
This encodes the prior "person left cam A → likely next seen on neighbor B".
"""

import numpy as np

CONFIRMED_THRESHOLD = 0.85
UNCERTAIN_THRESHOLD = 0.70
TOPOLOGY_BOOST = 0.03
TOPOLOGY_BOOST_CAP = 0.80  # boosted score never exceeds this


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    an = a / (np.linalg.norm(a) + 1e-8)
    bn = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(an, bn))


def find_best_match(
    embedding: np.ndarray,
    candidates: list[tuple[int, np.ndarray]],           # (global_id, emb)
    identity_cameras: dict[int, set[str]] | None = None,  # gid → seen cameras
    camera_id: str = "",
    topology: dict[str, set[str]] | None = None,         # camera → neighbors
    confirmed_threshold: float = CONFIRMED_THRESHOLD,
    uncertain_threshold: float = UNCERTAIN_THRESHOLD,
    topology_boost: float = TOPOLOGY_BOOST,
) -> tuple[int | None, float, bool]:
    """Best matching global_id for an embedding.

    Returns (global_id, score, confident).
    global_id is None if no candidate passes the uncertain threshold.
    """
    neighbors = topology.get(camera_id, set()) if topology else set()

    best_gid: int | None = None
    best_score = 0.0
    best_confident = True

    for gid, emb in candidates:
        sim = cosine_similarity(embedding, emb)
        if neighbors and identity_cameras:
            seen_cams = identity_cameras.get(gid, set())
            if seen_cams & neighbors:
                sim = min(sim + topology_boost, TOPOLOGY_BOOST_CAP)
        if sim > best_score:
            best_score = sim
            best_gid = gid

    if best_gid is None or best_score < uncertain_threshold:
        return None, best_score, False

    confident = best_score >= confirmed_threshold
    return best_gid, best_score, confident
