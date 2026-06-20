from __future__ import annotations

import numpy as np

from backend.detector.base_detector import BaseDetector
from backend.schemas import DetectionResult


class BabyDetector(BaseDetector):
    # 寶寶位置偵測預留模組。
    # 未來可接 YOLO、ONNX Runtime、MediaPipe、自訓練寶寶模型。

    name = "baby_detector"

    def detect(self, frame: np.ndarray) -> DetectionResult:
        return DetectionResult(
            detector_name=self.name,
            event_type="baby_position_warning",
            is_abnormal=False,
            confidence=0.0,
            message="寶寶位置偵測尚未啟用",
            extra={"enabled": False},
        )
