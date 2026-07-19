import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from src.db.schema import CREATE_TABLES


class AnalyticsDB:
    def __init__(self, db_path: str | Path = "outputs/analytics.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def connect(self):
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(CREATE_TABLES)
        self._conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        return self._conn

    # --- Cameras ---

    def upsert_camera(self, data: dict) -> str:
        cur = self.conn.execute(
            """INSERT INTO cameras (id, name, source, fps, resolution, location, gps, timezone, status)
               VALUES (:id, :name, :source, :fps, :resolution, :location, :gps, :timezone, :status)
               ON CONFLICT(id) DO UPDATE SET
               name=excluded.name, status=excluded.status, source=excluded.source""",
            data,
        )
        self.conn.commit()
        return data["id"]

    # --- Analysis Runs ---

    def start_run(self, camera_id: str, video_path: str, model_family: str, model_size: str) -> int:
        cur = self.conn.execute(
            """INSERT INTO analysis_runs (camera_id, video_path, model_family, model_size, started_at, status)
               VALUES (?, ?, ?, ?, datetime('now'), 'running')""",
            (camera_id, video_path, model_family, model_size),
        )
        self.conn.commit()
        return cur.lastrowid

    def finish_run(self, run_id: int, result: dict):
        self.conn.execute(
            """UPDATE analysis_runs SET
               total_frames=:tf, duration_sec=:ds, fps=:fps, resolution=:res,
               calibration=:cal, zones_config=:zc, completed_at=datetime('now'), status='completed'
               WHERE id=:id""",
            {
                "id": run_id,
                "tf": result.get("total_frames_processed"),
                "ds": result.get("video_duration_sec"),
                "fps": result.get("fps"),
                "res": result.get("resolution"),
                "cal": json.dumps(result.get("calibration", {})),
                "zc": json.dumps(result.get("zones", {})),
            },
        )
        self.conn.commit()

    # --- Tracks ---

    def insert_track(self, run_id: int, camera_id: str, obj: dict):
        path = obj.get("path", [])
        self.conn.execute(
            """INSERT INTO tracks (run_id, camera_id, track_id, class_name, class_id,
               first_frame, last_frame, duration_frames, distance_pixels, distance_meters,
               speed_mps, direction, state, confidence, path_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                camera_id,
                obj["id"],
                obj["class"],
                obj.get("class_id"),
                obj.get("first_frame"),
                obj.get("last_frame"),
                obj.get("duration_frames"),
                obj.get("movement", {}).get("distance_pixels"),
                obj.get("movement", {}).get("distance_meters"),
                obj.get("movement", {}).get("speed_mps"),
                obj.get("movement", {}).get("direction"),
                "ended",
                1.0,
                json.dumps(path),
            ),
        )
        self.conn.commit()

    def insert_tracks_batch(self, run_id: int, camera_id: str, objects: list[dict]):
        for obj in objects:
            self.insert_track(run_id, camera_id, obj)

    # --- Events ---

    def insert_event(self, run_id: int, camera_id: str, ev: dict, frame_index: int = 0, utc_time: str = ""):
        loc = ev.get("location", [])
        self.conn.execute(
            """INSERT INTO events (run_id, camera_id, event_type, track_id, class_name,
               zone, severity, confidence, duration_sec, location_x, location_y,
               message, utc_time, frame_index)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                camera_id,
                ev["type"],
                ev.get("track_id"),
                ev.get("class"),
                ev.get("zone", ""),
                ev.get("severity", "info"),
                ev.get("confidence", 1.0),
                ev.get("duration", 0),
                loc[0] if len(loc) > 0 else None,
                loc[1] if len(loc) > 1 else None,
                ev.get("message", ""),
                utc_time,
                frame_index,
            ),
        )
        self.conn.commit()

    def insert_events_batch(self, run_id: int, camera_id: str, events: list[dict], time_sync=None):
        for ev in events:
            self.insert_event(run_id, camera_id, ev)

    # --- Gate Counts ---

    def upsert_gate_count(self, run_id: int, camera_id: str, gate_name: str, entries: int, exits: int, net: int):
        self.conn.execute(
            """INSERT INTO gate_counts (run_id, camera_id, gate_name, entries, exits, net)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id, camera_id, gate_name) DO UPDATE SET
               entries=excluded.entries, exits=excluded.exits, net=excluded.net""",
            (run_id, camera_id, gate_name, entries, exits, net),
        )
        self.conn.commit()

    # --- Queries ---

    def query(self, sql: str, params: dict | tuple = ()) -> list[dict]:
        cur = self.conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def get_events(self, camera_id: str | None = None, severity: str | None = None, limit: int = 100) -> list[dict]:
        parts = ["SELECT * FROM events WHERE 1=1"]
        params = []
        if camera_id:
            parts.append("AND camera_id=?")
            params.append(camera_id)
        if severity:
            parts.append("AND severity=?")
            params.append(severity)
        parts.append("ORDER BY created_at DESC LIMIT ?")
        params.append(limit)
        return self.query(" ".join(parts), tuple(params))

    def get_tracks(self, run_id: int) -> list[dict]:
        return self.query("SELECT * FROM tracks WHERE run_id=?", {"run_id": run_id})
