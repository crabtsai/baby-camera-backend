from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from backend.alert_service import AlertService
from backend.config import Settings
from backend.database import Database
from backend.schemas import AlertPolicy, DetectionResult, EventRecord
from backend.services.snapshot_service import SnapshotService

logger = logging.getLogger(__name__)


@dataclass
class EventState:
    # 單一事件類型的執行中狀態；後端重啟後會重新計算。
    abnormal_since: float | None = None
    last_event_at: float = 0.0
    last_alert_at: float = 0.0
    alert_count_today: int = 0
    alert_day: str = ""
    occurrence_count: int = 0


class EventManager:
    # 將 DetectionResult 轉成正式事件。
    # Detector 只判斷「目前 frame 是否異常」。
    # EventManager 負責持續時間、冷卻時間、截圖、資料庫、告警。
    def __init__(
        self,
        settings: Settings,
        database: Database,
        snapshot_service: SnapshotService,
        alert_service: AlertService,
    ):
        self.settings = settings
        self.database = database
        self.snapshot_service = snapshot_service
        self.alert_service = alert_service
        self.states: dict[str, EventState] = {}
        self.alert_policies = self._build_alert_policies(settings)

    @staticmethod
    def _build_alert_policies(settings: Settings) -> dict[str, AlertPolicy]:
        # 告警策略表：要調整哪些事件會發 LINE、冷卻多久、一天最多幾次，都改這裡。
        return {
            "room_too_dark": AlertPolicy(
                # 畫面太暗可能持續很久，所以用每日上限避免 LINE 一直洗訊息。
                duration_sec=settings.dark_duration_sec,
                cooldown_sec=settings.dark_cooldown_sec,
                level="warning",
                notify_line=True,
                max_alerts_per_day=10,
            ),
            "low_motion_warning": AlertPolicy(
                # 長時間低動作目前先只留紀錄，不主動發 LINE。
                duration_sec=settings.motion_duration_sec,
                cooldown_sec=settings.motion_cooldown_sec,
                level="info",
                notify_line=False,
            ),
            "baby_position_warning": AlertPolicy(
                # 寶寶位置異常需要累積到指定次數後才發 LINE，降低誤報。
                duration_sec=5,
                cooldown_sec=300,
                level="critical",
                notify_line=True,
                alert_after_count=1,  # 例：第 1 次只記錄，第 2 次才通知 LINE。
            ),
            "face_cover_warning": AlertPolicy(
                # 臉部遮蔽風險較高，通過持續時間條件後就通知 LINE。
                duration_sec=5,
                cooldown_sec=180,
                level="critical",
                notify_line=True,
            ),
            "test_alert": AlertPolicy(duration_sec=0, cooldown_sec=0, level="critical"),
        }

    def handle_results(self, frame: np.ndarray, results: list[DetectionResult]) -> list[EventRecord]:
        events: list[EventRecord] = []
        for result in results:
            event = self.handle_result(frame, result)
            if event:
                events.append(event)
        return events

    def handle_result(self, frame: np.ndarray, result: DetectionResult) -> EventRecord | None:
        event_type = result.event_type
        policy = self.alert_policies.get(event_type, AlertPolicy(duration_sec=5, cooldown_sec=300))
        state = self.states.setdefault(event_type, EventState())
        now = time.monotonic()

        # 偵測結果恢復正常時，異常計時歸零。
        if not result.is_abnormal:
            state.abnormal_since = None
            return None

        # 第一次異常只開始計時；超過 duration_sec 才真正建立事件。
        if state.abnormal_since is None:
            state.abnormal_since = now
            logger.info("Abnormal started: %s", event_type)
            return None

        abnormal_duration = now - state.abnormal_since
        if abnormal_duration < policy.duration_sec:
            return None

        # 冷卻時間用來限制事件建立頻率，避免同一狀況把資料庫塞滿。
        if now - state.last_event_at < policy.cooldown_sec:
            return None

        # 只計算「已成立的事件」，不是每一張異常 frame 都加一。
        state.occurrence_count += 1

        snapshot_path = self.snapshot_service.save_snapshot(frame, result)
        should_alert = self._should_alert(state, policy)

        # 已成立事件一定寫入資料庫；LINE 是否發送由策略決定。
        event = EventRecord(
            camera_id=self.settings.camera_id,
            event_type=result.event_type,
            detector_name=result.detector_name,
            message=result.message,
            confidence=float(result.confidence),
            snapshot_path=snapshot_path,
            extra={
                **result.extra,
                "abnormal_duration_sec": round(abnormal_duration, 2),
                "cooldown_sec": policy.cooldown_sec,
                "level": policy.level,
                "notify_line": should_alert,
                "occurrence_count": state.occurrence_count,
                "alert_after_count": policy.alert_after_count,
                "max_alerts_per_day": policy.max_alerts_per_day,
            },
        )

        event = self.database.insert_event(event)
        state.last_event_at = now
        if should_alert:
            # 只有通過策略檢查後才會真正發 LINE。
            self.alert_service.send_alert(event, policy)
            state.last_alert_at = now
            state.alert_count_today += 1
        logger.warning("Event created: %s", event)
        return event

    @staticmethod
    def _should_alert(state: EventState, policy: AlertPolicy) -> bool:
        # 每日上限用本機日期計算，隔天會重新允許發送。
        today = datetime.now().date().isoformat()
        if state.alert_day != today:
            state.alert_day = today
            state.alert_count_today = 0

        if not policy.notify_line:
            return False

        # 例：alert_after_count=2 代表第 1 次只記錄，第 2 次才通知。
        if state.occurrence_count < policy.alert_after_count:
            return False

        if policy.max_alerts_per_day is not None:
            return state.alert_count_today < policy.max_alerts_per_day

        return True

    def create_test_event(self) -> EventRecord:
        policy = self.alert_policies["test_alert"]
        event = EventRecord(
            camera_id=self.settings.camera_id,
            event_type="test_alert",
            detector_name="manual_test",
            message="這是一則寶寶攝影機測試告警",
            confidence=1.0,
            snapshot_path=None,
            extra={"source": "POST /test-alert"},
        )
        event = self.database.insert_event(event)
        self.alert_service.send_alert(event, policy)
        return event
