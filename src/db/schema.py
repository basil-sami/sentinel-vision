CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS cameras (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    source      TEXT NOT NULL,
    fps         INTEGER DEFAULT 25,
    resolution  TEXT DEFAULT '640,360',
    location    TEXT DEFAULT '',
    gps         TEXT,
    timezone    TEXT DEFAULT 'UTC',
    status      TEXT DEFAULT 'offline',
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id       TEXT REFERENCES cameras(id),
    video_path      TEXT,
    model_family    TEXT,
    model_size      TEXT,
    total_frames    INTEGER,
    duration_sec    REAL,
    fps             REAL,
    resolution      TEXT,
    started_at      TEXT,
    completed_at    TEXT,
    calibration     TEXT,
    zones_config    TEXT,
    status          TEXT DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS tracks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES analysis_runs(id),
    camera_id       TEXT REFERENCES cameras(id),
    track_id        INTEGER NOT NULL,
    class_name      TEXT NOT NULL,
    class_id        INTEGER,
    first_frame     INTEGER,
    last_frame      INTEGER,
    duration_frames INTEGER,
    distance_pixels REAL,
    distance_meters REAL,
    speed_mps       REAL,
    direction       TEXT,
    state           TEXT DEFAULT 'active',
    confidence      REAL DEFAULT 1.0,
    appearance_var  REAL DEFAULT 0.0,
    occlusion_count INTEGER DEFAULT 0,
    path_json       TEXT,
    UNIQUE(run_id, track_id)
);

CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES analysis_runs(id),
    camera_id       TEXT REFERENCES cameras(id),
    event_type      TEXT NOT NULL,
    track_id        INTEGER,
    class_name      TEXT,
    zone            TEXT,
    severity        TEXT DEFAULT 'info',
    confidence      REAL DEFAULT 1.0,
    duration_sec    REAL,
    location_x      INTEGER,
    location_y      INTEGER,
    message         TEXT,
    incident_id     INTEGER,
    utc_time        TEXT,
    frame_index     INTEGER,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS incidents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES analysis_runs(id),
    camera_id       TEXT REFERENCES cameras(id),
    incident_type   TEXT NOT NULL,
    severity        TEXT DEFAULT 'medium',
    status          TEXT DEFAULT 'open',
    first_event_at  TEXT,
    last_event_at   TEXT,
    summary         TEXT,
    resolved_at     TEXT
);

CREATE TABLE IF NOT EXISTS gate_counts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES analysis_runs(id),
    camera_id       TEXT REFERENCES cameras(id),
    gate_name       TEXT NOT NULL,
    entries         INTEGER DEFAULT 0,
    exits           INTEGER DEFAULT 0,
    net             INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS evidence_clips (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES analysis_runs(id),
    camera_id       TEXT REFERENCES cameras(id),
    event_id        INTEGER REFERENCES events(id),
    clip_path       TEXT NOT NULL,
    metadata_json   TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity);
CREATE INDEX IF NOT EXISTS idx_events_incident ON events(incident_id);
CREATE INDEX IF NOT EXISTS idx_tracks_run ON tracks(run_id);
"""
