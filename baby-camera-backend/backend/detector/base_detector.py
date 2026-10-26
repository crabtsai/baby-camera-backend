from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from backend.schemas import DetectionResult


class BaseDetector(ABC):
    # 所有 detector 的共同介面。
    # 只要實作 detect(frame)，DetectorManager 就能統一呼叫。
    name = "base_detector"

    def reset(self) -> None:
        """Clear temporal state when the monitored subject is unavailable."""

    def close(self) -> None:
        """Release native model resources after inference stops."""

    def detect_many(self, frame: np.ndarray) -> list[DetectionResult]:
        """Allow a shared inference pipeline to emit several independent events."""
        return [self.detect(frame)]

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        # 輸入 OpenCV frame，輸出標準 DetectionResult。
        raise NotImplementedError
