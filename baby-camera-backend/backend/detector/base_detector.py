from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from backend.schemas import DetectionResult


class BaseDetector(ABC):
    name = "base_detector"

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        # 輸入 OpenCV frame，輸出標準 DetectionResult。
        raise NotImplementedError
