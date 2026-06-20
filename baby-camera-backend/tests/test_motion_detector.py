import numpy as np

from backend.detector.motion_detector import MotionDetector


def test_motion_detector_initial_frame_is_not_abnormal():
    detector = MotionDetector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detector.detect(frame)
    assert result.is_abnormal is False
