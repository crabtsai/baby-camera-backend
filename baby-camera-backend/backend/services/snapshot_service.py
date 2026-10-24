from __future__ import annotations

import logging
import re
from pathlib import Path

import cv2
import numpy as np

from backend.schemas import DetectionResult
from backend.utils.time_utils import timestamp_for_filename

logger = logging.getLogger(__name__)


class SnapshotService:
    # 事件成立時保存當下畫面，方便回頭確認告警發生的實際影像。
    def __init__(self, snapshot_dir: Path):
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_name(value: str) -> str:
        # camera_id 可能來自 config.ini，轉成檔名安全字元，避免路徑穿越或特殊符號問題。
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unknown"

    def save_snapshot(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        camera_id: str | None = None,
    ) -> str | None:
        # 多相機時用 camera_id 分資料夾，方便直接從檔案系統回查是哪台相機觸發。
        camera_dir = self.snapshot_dir / self._safe_name(camera_id or "default")
        camera_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{timestamp_for_filename()}_{self._safe_name(result.event_type)}.jpg"
        path = camera_dir / filename
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            logger.warning("Failed to save snapshot: %s", path)
            return None
        return str(path)
