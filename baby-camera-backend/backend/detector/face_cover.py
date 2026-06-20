from __future__ import annotations

import numpy as np

from backend.detector.base_detector import BaseDetector
from backend.schemas import DetectionResult


class FaceCoverDetector(BaseDetector):
    # 臉部遮蔽偵測預留模組。
    # 未來可接臉部偵測、影像分割、YOLO-seg、趴睡 / 棉被遮臉判斷。

    name = "face_cover_detector"

    def detect(self, frame: np.ndarray) -> DetectionResult:
        return DetectionResult(
            detector_name=self.name,
            event_type="face_cover_warning",
            is_abnormal=False,
            confidence=0.0,
            message="臉部遮蔽偵測尚未啟用",
            extra={"enabled": False},
        )
