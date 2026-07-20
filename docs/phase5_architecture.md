# Phase 5 — Multi-Camera Intelligence & Evidence Management

These are the two hardest remaining engineering problems. They separate a **good CCTV analyzer** from a **modern VMS (Video Management System)**.

---

## Part 1 — Multi-Camera Intelligence

### Current Situation

Each camera operates independently. The same person walking through 4 cameras gets 4 different track IDs. The system has no way of knowing Track 17 in Camera A is the same human as Track 4 in Camera B and Track 12 in Camera C.

### Goal

A single **Global Person ID** that follows the subject across all cameras:

```
Camera A              Camera B              Camera C
Track 17  ──►  Global 1234  ◄──  Track 4  ◄──  Track 12
```

Now the operator searches `Global Person 1234` instead of `Track 17`.

---

### Step 1 — Tracklets

Every local tracker becomes a producer. Instead of raw detections, it exports **Tracklets** — a complete record of what a single camera observed:

```json
{
    "camera": "gate_north",
    "track": 17,
    "start": "12:01:04",
    "end": "12:02:30",
    "appearance": [0.13, 0.88, ...],
    "trajectory": [...],
    "bbox": [...],
    "class": "person"
}
```

A Tracklet is: "Camera X observed something from time T1 to T2."

---

### Step 2 — Global Identity Manager

New module:

```
src/identity/
    global_identity.py    # GlobalIdentityManager
    matcher.py            # Appearance + spatiotemporal matcher
    graph.py              # Camera topology graph
    topology.py           # Travel time / reachability
    timeline.py           # Global timeline builder
```

Its job:

```
Receive Tracklets  →  Compare  →  Merge  →  Assign Global IDs
```

---

### Step 3 — Appearance Matching

Use the existing OSNet ReID model. Instead of comparing every frame, compare a **single average embedding** per track:

```
Track 17:  [0.11, 0.32, 0.51, ...]   ← average embedding
Track 41:  [0.12, 0.31, 0.50, ...]   ← average embedding
Similarity: 98%  →  probable same person
```

---

### Step 4 — Camera Topology

Never compare every camera pair equally. Model the site as a directed graph:

```
North Gate  ──►  Lobby  ──►  Hallway  ──►  Vault
                              │
                              ▼
                          Parking Lot
```

A person leaving the Vault cannot instantly appear in the Parking Lot. Topology eliminates impossible matches.

```python
@dataclass
class CameraNode:
    name: str
    neighbors: dict[str, float]  # neighbor_name → travel_time_sec
    direction: str               # "one-way" | "two-way"
```

---

### Step 5 — Time Constraints

If Camera A sees a person at 12:00 and Camera B (400m away) sees a person at 12:00:02, the match is impossible. Reject.

Travel time between camera pairs defines the valid time window for a match.

---

### Step 6 — Identity Score

Instead of binary match/no-match, compute a weighted score:

```
IdentityScore =
    w1 × Appearance       (cosine similarity of embeddings)
  + w2 × Topology         (reachability in camera graph)
  + w3 × TravelTime       (plausible transit duration)
  + w4 × Motion           (direction consistency)
  + w5 × Class            (person / vehicle / object)
  + w6 × Confidence       (track detection confidence)
```

Default weights: `Appearance=0.50`, `Topology=0.20`, `TravelTime=0.20`, `Motion=0.10`.

---

### Step 7 — Global Timeline

Instead of isolated tracks, build a per-identity journal:

```
Global Person 1234
  12:01  Gate (north entrance)
  12:04  Lobby
  12:09  Office corridor
  12:22  Parking lot (exit)
```

This is what security operators actually need.

---

### Vehicles

Vehicles are easier — their identity key is:

```
Plate + Color + Model + Dimensions + Appearance embedding
```

When plate OCR fails, the system still tracks by appearance + attributes.

---

### Database

New table:

```sql
CREATE TABLE global_identities (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,          -- 'person' | 'vehicle' | 'object'
    confidence REAL,
    first_seen TEXT,
    last_seen TEXT,
    camera_count INTEGER,
    tracklet_ids TEXT            -- JSON list
);

CREATE TABLE tracklets (
    id INTEGER PRIMARY KEY,
    global_id INTEGER REFERENCES global_identities(id),
    camera TEXT,
    local_track_id INTEGER,
    start_time TEXT,
    end_time TEXT,
    embedding BLOB,
    trajectory TEXT,             -- JSON
    bbox_sequence TEXT           -- JSON
);
```

---

## Part 2 — Evidence Management

This is almost a separate product. Every incident should produce a self-contained **Evidence Package**.

### Current

```
clip.mp4
```

That's not enough.

### Target

Every incident becomes an **Evidence Bundle**:

```
incident_00042/
    metadata.json
    raw.mp4
    annotated.mp4
    thumbnail.jpg
    tracks.json
    events.json
    zones.json
    timeline.json
    hashes.json
```

---

### metadata.json

```json
{
    "incident_id": 42,
    "severity": "high",
    "camera": "gate_north",
    "utc_time": "2026-07-20T12:01:04Z",
    "operator": "system",
    "hash_algorithm": "SHA-256",
    "global_ids": [1234],
    "vehicle_ids": ["ABC123"]
}
```

### tracks.json

Every object involved, with full trajectory, speed, and identity:

```json
[
    {
        "global_id": 1234,
        "local_tracks": [{"camera": "gate_north", "track": 17}],
        "trajectory": [[100, 200], [105, 205], ...],
        "speed_mps": 1.2,
        "class": "person"
    }
]
```

### events.json

All events in temporal order:

```json
[
    {"time": "12:01:04", "type": "zone_entry", "detail": "North Gate"},
    {"time": "12:05:12", "type": "possible_loitering", "detail": "Lobby, 240s dwell"},
    {"time": "12:09:33", "type": "abandoned_object", "detail": "Backpack dropped"},
    {"time": "12:11:05", "type": "zone_exit", "detail": "North Gate"},
    {"time": "12:16:00", "type": "security_alert", "detail": "Dispatch notified"}
]
```

### timeline.json

Human-readable incident timeline:

```
12:01  Person enters via North Gate
12:05  Loitering detected in Lobby
12:09  Backpack dropped near bench
12:11  Person exits via North Gate
12:16  Security dispatched
```

---

### Evidence Clips

Instead of isolating the event frame, store:

```
30 seconds before event  +  event duration  +  30 seconds after event
```

Operators almost always need context.

---

### Snapshot System

Automatically save:

- **First frame** — when the subject enters scene
- **Peak event frame** — the moment of highest severity (e.g., object drop, weapon detection)
- **Last frame** — when the subject leaves

---

### Integrity (Chain of Custody)

Every evidence package contains a `hashes.json`:

```json
{
    "raw.mp4": "sha256:e3b0c44298fc1c149afbf4c8996fb924...",
    "annotated.mp4": "sha256:d7a8fbb307d7809469ca9abcb0082e4f...",
    "metadata.json": "sha256:9f86d081884c7d659a2feaa0c55ad015...",
    "tracks.json": "sha256:6b23c0d5f35d1b11f9b68a0b1b0e5c8b..."
}
```

If any file is modified after creation, verification fails. Critical for **chain-of-custody** and **audit** purposes.

---

### Evidence Index

Database table for discovery:

```sql
CREATE TABLE evidence_packages (
    id INTEGER PRIMARY KEY,
    incident_id INTEGER REFERENCES incidents(id),
    storage_path TEXT,
    hash_sha256 TEXT,
    severity TEXT,
    global_ids TEXT,        -- JSON list for search
    vehicle_ids TEXT,       -- JSON list for search
    camera TEXT,
    utc_time TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

Operators can now search:

- `Person 1234 → all incidents + evidence`
- `Plate ABC123 → all evidence packages`
- `North Gate → all clips from today`

Without opening videos manually.

---

## Proposed Module Structure

```
src/
├── identity/
│   ├── __init__.py
│   ├── global_identity.py    # GlobalIdentityManager
│   ├── matcher.py            # Appearance + spatiotemporal matching
│   ├── graph.py              # CameraNode, TopologyGraph
│   ├── topology.py           # Travel time / reachability constraints
│   └── timeline.py           # Global timeline builder
│
├── evidence/
│   ├── __init__.py
│   ├── bundle.py             # EvidenceBundle dataclass
│   ├── recorder.py           # Clip capture with context window
│   ├── snapshot.py           # First/peak/last frame capture
│   ├── hasher.py             # SHA-256 integrity verification
│   ├── exporter.py           # Export to directory / archive
│   └── archive.py            # Cleanup / retention policy
│
├── database/
│   ├── identities.py         # Global identity CRUD
│   ├── incidents.py          # Incident record CRUD
│   └── evidence.py           # Evidence package index CRUD
```

---

## Implementation Strategy

**Do not** try to make multi-camera fusion work with arbitrary cameras as the first implementation.

Start with a **fixed set of synchronized cameras** covering one facility, where the topology and approximate travel times are known in advance. Once that works reliably, gradually relax assumptions.

Incremental approach:

1. **Phase 5a**: Tracklet export + GlobalIdentityManager with appearance-only matching (single topology)
2. **Phase 5b**: Camera topology graph + time constraints → weighted identity score
3. **Phase 5c**: Global timeline builder + database tables
4. **Phase 5d**: Evidence Bundle format + recorder with context window
5. **Phase 5e**: Snapshot system + SHA-256 integrity
6. **Phase 5f**: Evidence index + search API

This avoids the trap of trying to solve the fully general problem from the outset.
