from __future__ import annotations

import numpy as np

from backend.detector.base_detector import BaseDetector
from backend.schemas import DetectionResult
from backend.utils.image_utils import to_gray


class DarkDetector(BaseDetector):
    # 畫面過暗偵測，不需要 AI 模型，只用平均亮度判斷。
    name = "dark_detector"

    def __init__(self, brightness_threshold: float = 30):
        # brightness_threshold 越高，越容易判斷為太暗。
        self.brightness_threshold = brightness_threshold

    def detect(self, frame: np.ndarray) -> DetectionResult:
        # 用灰階平均亮度判斷畫面是否過暗。
        gray = to_gray(frame)
        brightness = float(gray.mean())
        is_dark = brightness < self.brightness_threshold

        if is_dark:
            confidence = min(
                1.0,
                max(0.0, (self.brightness_threshold - brightness) / self.brightness_threshold),
            )
        else:
            confidence = 0.0

        return DetectionResult(
            detector_name=self.name,
            event_type="room_too_dark",
            is_abnormal=is_dark,
            confidence=confidence,
            message="畫面太暗，請確認房間照明或攝影機遮蔽" if is_dark else "畫面亮度正常",
            extra={
                "brightness": brightness,
                "threshold": self.brightness_threshold,
            },
        )
