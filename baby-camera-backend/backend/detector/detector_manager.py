from __future__ import annotations

import logging

import numpy as np

from backend.config import Settings
from backend.detector.baby_detector import BabyDetector
from backend.detector.base_detector import BaseDetector
from backend.detector.dark_detector import DarkDetector
from backend.detector.face_cover import FaceCoverDetector
from backend.detector.motion_detector import MotionDetector
from backend.schemas import DetectionResult

logger = logging.getLogger(__name__)


class DetectorManager:
    def __init__(self, detectors: list[BaseDetector]):
        self.detectors = detectors
        logger.info("Enabled detectors: %s", [d.name for d in detectors])

    @classmethod
    def from_settings(cls, settings: Settings) -> "DetectorManager":
        detectors: list[BaseDetector] = []

        if settings.enable_dark_detector:
            detectors.append(DarkDetector(settings.dark_brightness_threshold))

        if settings.enable_motion_detector:
            detectors.append(
                MotionDetector(
                    diff_threshold=settings.motion_diff_threshold,
                    low_ratio_threshold=settings.motion_low_ratio_threshold,
                )
            )

        if settings.enable_baby_detector:
            detectors.append(BabyDetector())

        if settings.enable_face_cover_detector:
            detectors.append(FaceCoverDetector())

        return cls(detectors)

    def run_all(self, frame: np.ndarray) -> list[DetectionResult]:
        results: list[DetectionResult] = []
        for detector in self.detectors:
            try:
                results.append(detector.detect(frame))
            except Exception:
                logger.exception("Detector failed: %s", detector.name)
        return results

    def names(self) -> list[str]:
        return [detector.name for detector in self.detectors]
