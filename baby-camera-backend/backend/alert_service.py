from __future__ import annotations

import logging

from backend.config import Settings
from backend.schemas import AlertPolicy, EventRecord
from backend.services.line_service import LineService

logger = logging.getLogger(__name__)


class AlertService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.line_service = LineService(settings)

    def send_alert(self, event: EventRecord, policy: AlertPolicy) -> bool:
        # 有些事件只適合留紀錄，太頻繁時不適合一直傳 LINE。
        if not policy.notify_line:
            logger.info("Event recorded without LINE notification. event_type=%s", event.event_type)
            return False

        # AlertService 是通知入口；Firebase HTTP 細節交給 LineService 處理。
        return self.line_service.push_event(
            event=event,
            level=policy.level,
            notify_line=policy.notify_line,
        )
