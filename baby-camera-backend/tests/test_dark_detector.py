import numpy as np

from backend.detector.dark_detector import DarkDetector


def test_dark_detector_detects_dark_frame():
    detector = DarkDetector(brightness_threshold=30)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detector.detect(frame)
    assert result.is_abnormal is True
    assert result.event_type == "room_too_dark"
