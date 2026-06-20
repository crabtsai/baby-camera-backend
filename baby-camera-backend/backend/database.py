from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from backend.schemas import EventRecord


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    detector_name TEXT NOT NULL,
                    message TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    snapshot_path TEXT,
                    extra_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)"
            )

    def insert_event(self, event: EventRecord) -> EventRecord:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO events (
                    camera_id,
                    event_type,
                    detector_name,
                    message,
                    confidence,
                    snapshot_path,
                    extra_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.camera_id,
                    event.event_type,
                    event.detector_name,
                    event.message,
                    float(event.confidence),
                    event.snapshot_path,
                    json.dumps(event.extra, ensure_ascii=False),
                    event.created_at,
                ),
            )
            event.id = int(cursor.lastrowid)
            return event

    def list_events(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["extra"] = json.loads(item.pop("extra_json") or "{}")
            except json.JSONDecodeError:
                item["extra"] = {}
            result.append(item)
        return result
