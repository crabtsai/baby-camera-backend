import numpy as np

from backend.detector.prone_sleep_detector import ProneSleepDetector


def _person(confidences: dict[int, float]) -> dict:
    keypoints = [{"x": 0, "y": 0, "confidence": 0.0} for _ in range(17)]
    for index, confidence in confidences.items():
        keypoints[index] = {"x": float(index), "y": float(index), "confidence": confidence}
    return {"keypoints": keypoints}


def test_reports_prone_risk_when_body_is_visible_but_face_is_not():
    detector = ProneSleepDetector(
        predict_fn=lambda frame: [_person({5: 0.9, 6: 0.9, 11: 0.8, 12: 0.8})]
    )

    result = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert result.is_abnormal is True
    assert result.event_type == "prone_sleep_warning"
    assert result.extra["assessments"][0]["face_down_risk"] is True


def test_stays_normal_when_face_and_body_are_visible():
    visible = {index: 0.9 for index in (0, 1, 2, 3, 4, 5, 6, 11, 12)}
    detector = ProneSleepDetector(predict_fn=lambda frame: [_person(visible)])

    result = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert result.is_abnormal is False
    assert result.extra["assessments"][0]["visible_face_keypoints"] == 5


def test_does_not_warn_when_pose_is_too_incomplete_to_assess():
    detector = ProneSleepDetector(
        predict_fn=lambda frame: [_person({5: 0.9, 6: 0.9})],
        min_body_keypoints=3,
    )

    result = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert result.is_abnormal is False
    assert result.extra["assessments"][0]["body_visible"] is False


def test_missing_model_disables_detector_safely():
    detector = ProneSleepDetector(model_path="")

    result = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert result.is_abnormal is False
    assert result.extra["enabled"] is False
