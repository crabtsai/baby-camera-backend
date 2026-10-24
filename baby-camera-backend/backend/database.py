from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from backend.schemas import EventRecord


class Database:
    # SQLite 儲存層：目前只負責事件紀錄，之後可再擴充設定或統計表。
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.write_lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        # row_factory 讓查詢結果可以像 dict 一樣用欄位名稱取值。
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        # 多相機 worker 可能同時寫入 SQLite，busy_timeout 可降低 database locked 機率。
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    def init_db(self) -> None:
        # 首次啟動自動建立資料表與索引，避免需要手動 migration 才能開發測試。
        with self._connect() as conn:
            # WAL 比預設 journal 更適合讀寫並行，對多相機事件寫入比較穩。
            conn.execute("PRAGMA journal_mode = WAL")
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_camera_id ON events(camera_id)"
            )

    def insert_event(self, event: EventRecord) -> EventRecord:
        # extra 可能包含 detector 的詳細判斷資料，因此用 JSON 存進單一欄位。
        for attempt in range(3):
            try:
                with self.write_lock:
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
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 2:
                    raise
                time.sleep(0.15 * (attempt + 1))
        return event

    def list_events(self, limit: int = 50, camera_id: str | None = None) -> list[dict[str, Any]]:
        # 限制最多 500 筆，避免 API 一次撈太多資料拖慢頁面。
        limit = max(1, min(limit, 500))
        with self._connect() as conn:
            if camera_id:
                rows = conn.execute(
                    """
                    SELECT * FROM events
                    WHERE camera_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (camera_id, limit),
                ).fetchall()
            else:
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
                # 將 DB 內的 extra_json 還原成前端容易使用的 extra dict。
                item["extra"] = json.loads(item.pop("extra_json") or "{}")
            except json.JSONDecodeError:
                item["extra"] = {}
            result.append(item)
        return result
