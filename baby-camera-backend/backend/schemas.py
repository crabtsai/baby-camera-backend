from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class DetectionResult:
    detector_name: str
    event_type: str
    is_abnormal: bool
    confidence: float
    message: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventRecord:
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
