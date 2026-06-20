from __future__ import annotations

import logging
from typing import Any

import requests

from backend.config import Settings
from backend.schemas import EventRecord

logger = logging.getLogger(__name__)


class LineService:
    # Python 只負責把告警送到 Firebase Cloud Function；Firestore / LINE 由 Firebase 處理。
    def __init__(self, settings: Settings):
        self.settings = settings

    def push_event(self, event: EventRecord, level: str, notify_line: bool = True) -> bool:
        # 建立和 push_firebase.py 相同的 payload，確保手動測試與後端流程一致。
        payload = self._build_payload(event, level=level, notify_line=notify_line)

        # 模擬模式用來調整偵測器，不會真的發送 LINE。
        if self.settings.alert_dry_run:
            logger.info("ALERT_DRY_RUN=true, skip Firebase alert. payload=%s", payload)
            return True

        # 沒有設定 Cloud Function URL 時，事件仍會寫入資料庫，但不呼叫外部服務。
        if not self.settings.alert_webhook_url:
            logger.info("ALERT_WEBHOOK_URL is empty, alert only logged. payload=%s", payload)
            return False

        # Firebase Cloud Function 會用這個密鑰判斷請求是不是合法來源。
        headers = {"Content-Type": "application/json"}
        if self.settings.alert_ingest_key:
            headers["x-alert-key"] = self.settings.alert_ingest_key

        try:
            # LINE token 留在 Firebase；Python 只呼叫這個可信任的告警入口。
            resp = requests.post(
                self.settings.alert_webhook_url,
                json=payload,
                headers=headers,
                timeout=self.settings.alert_timeout_sec,
            )
            resp.raise_for_status()
            logger.info("Firebase alert sent: status=%s", resp.status_code)
            return True
        except Exception:
            logger.exception("Failed to send Firebase alert")
            return False

    @staticmethod
    def _build_payload(event: EventRecord, level: str, notify_line: bool) -> dict[str, Any]:
        # 上層欄位保持簡潔穩定；後端細節統一放在 extra。
        return {
            "version": 1,
            "cameraId": event.camera_id,
            "eventType": event.event_type,
            "level": level,
            "message": event.message,
            "score": event.confidence,
            "notifyLine": notify_line,
            "extra": {
                **event.extra,
                "detectorName": event.detector_name,
                "snapshotPath": event.snapshot_path,
                "createdAt": event.created_at,
                "eventId": event.id,
            },
        }
