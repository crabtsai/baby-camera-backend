import numpy as np

from backend.detector.motion_detector import MotionDetector


def test_motion_detector_initial_frame_is_not_abnormal():
    # 第一張 frame 只會建立背景參考，不應該立刻判斷為低動作異常。
    detector = MotionDetector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detector.detect(frame)
    assert result.is_abnormal is False
