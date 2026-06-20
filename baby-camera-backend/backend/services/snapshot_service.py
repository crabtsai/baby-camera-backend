from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from backend.schemas import DetectionResult
from backend.utils.time_utils import timestamp_for_filename

logger = logging.getLogger(__name__)


class SnapshotService:
    def __init__(self, snapshot_dir: Path):
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(self, frame: np.ndarray, result: DetectionResult) -> str | None:
        filename = f"{timestamp_for_filename()}_{result.event_type}.jpg"
        path = self.snapshot_dir / filename
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            logger.warning("Failed to save snapshot: %s", path)
            return None
        return str(path)
