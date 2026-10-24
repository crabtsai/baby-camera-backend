from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def now_iso() -> str:
    # 資料庫與告警 payload 使用秒級 ISO 時間，方便人讀也方便排序。
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class DetectionResult:
    # Detector 每次只回傳「單張 frame 的判斷結果」。
    # 後續是否建立事件、是否通知 LINE，交給 EventManager 決定。
    detector_name: str
    event_type: str
    is_abnormal: bool
    confidence: float
    message: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventRecord:
    # 已成立的事件紀錄，可寫入 SQLite，也可送往 Firebase Cloud Function。
    camera_id: str
    event_type: str
    message: str
    confidence: float
    detector_name: str
    snapshot_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso)
    id: int | None = None


@dataclass
class DetectorRule:
    # 保留給較簡單的 detector 規則使用；目前主要告警策略改由 AlertPolicy 控制。
    duration_sec: float
    cooldown_sec: float


@dataclass(frozen=True)
class AlertPolicy:
    # 定義單一事件類型要怎麼記錄，以及是否需要通知 LINE。
    duration_sec: float
    cooldown_sec: float
    level: str = "warning"
    notify_line: bool = True
    alert_after_count: int = 1
    max_alerts_per_day: int | None = None
