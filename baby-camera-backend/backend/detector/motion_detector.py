from __future__ import annotations

import cv2
import numpy as np

from backend.detector.base_detector import BaseDetector
from backend.schemas import DetectionResult
from backend.utils.image_utils import to_gray


class MotionDetector(BaseDetector):
    # 簡易低動作提醒。
    # 注意：這不是呼吸偵測，也不是醫療判斷。
    # 只是用 frame difference 估算畫面變化量。

    name = "motion_detector"

    def __init__(self, diff_threshold: float = 8, low_ratio_threshold: float = 0.003):
        self.diff_threshold = diff_threshold
        self.low_ratio_threshold = low_ratio_threshold
        self.prev_gray: np.ndarray | None = None

    def detect(self, frame: np.ndarray) -> DetectionResult:
        gray = to_gray(frame)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return DetectionResult(
                detector_name=self.name,
                event_type="low_motion_warning",
                is_abnormal=False,
                confidence=0.0,
                message="正在建立動作偵測基準畫面",
                extra={"motion_ratio": None},
            )

        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray

        _, binary = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        motion_pixels = int(np.count_nonzero(binary))
        total_pixels = int(binary.size)
        motion_ratio = motion_pixels / max(1, total_pixels)

        is_low_motion = motion_ratio < self.low_ratio_threshold
        if is_low_motion:
            confidence = 1.0 - min(1.0, motion_ratio / max(self.low_ratio_threshold, 1e-9))
        else:
            confidence = 0.0

        return DetectionResult(
            detector_name=self.name,
            event_type="low_motion_warning",
            is_abnormal=is_low_motion,
            confidence=confidence,
            message="畫面長時間變化偏低，請留意寶寶狀態" if is_low_motion else "畫面有正常變化",
            extra={
                "motion_ratio": motion_ratio,
                "low_ratio_threshold": self.low_ratio_threshold,
                "diff_threshold": self.diff_threshold,
            },
        )
