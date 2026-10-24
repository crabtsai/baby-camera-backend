import numpy as np

from backend.detector.baby_detector import BabyDetector


def make_frame():
    # 測試只驗證 YOLO 結果判斷邏輯，影像內容本身不重要。
    return np.zeros((32, 32, 3), dtype=np.uint8)


def test_baby_detector_is_normal_when_baby_face_is_visible():
    # 模擬 YOLO 回傳 baby_face，確認 detector 會判斷為正常。
    detector = BabyDetector(
        predict_fn=lambda frame: [
            {"label": "baby_face", "confidence": 0.85},
        ]
    )

    result = detector.detect(make_frame())

    assert result.is_abnormal is False
    assert result.message == "已偵測到寶寶臉部"
    assert result.extra["baby_face_count"] == 1


def test_baby_detector_alerts_when_baby_face_is_missing():
    # 模擬 YOLO 只看到玩具，確認沒有寶寶臉部時會產生位置告警。
    detector = BabyDetector(
        predict_fn=lambda frame: [
            {"label": "toy", "confidence": 0.9},
        ]
    )

    result = detector.detect(make_frame())

    assert result.is_abnormal is True
    assert result.event_type == "baby_position_warning"
    assert result.extra["baby_face_count"] == 0


def test_baby_detector_is_disabled_without_model_or_predict_fn():
    # 沒有模型也沒有測試用 predict_fn 時，detector 應安全停用而不是丟例外。
    detector = BabyDetector()

    result = detector.detect(make_frame())

    assert result.is_abnormal is False
    assert result.extra["enabled"] is False
