import numpy as np
import pytest
from backend.detector.face_cover import FaceCoverDetector


def item(label, box, confidence=.9):
    return {"label": label, "box": box, "confidence": confidence}

FACE = item("face", [10, 10, 90, 90])
NOSE = item("nose", [40, 35, 55, 50])
MOUTH = item("mouth", [35, 60, 65, 75])
PACIFIER = item("pacifier_in_mouth", [30, 57, 70, 80])


def run(*detections):
    return FaceCoverDetector(predict_fn=lambda _: list(detections)).detect(np.zeros((100, 100, 3), dtype=np.uint8))


def test_visible_parts_clear():
    assert run(FACE, NOSE, MOUTH).extra["assessment"] == "clear"


def test_missing_mouth_unknown_not_cover_evidence():
    r = run(FACE, NOSE)
    assert not r.is_abnormal and not r.extra["evaluable"]
    assert r.extra["missing_parts"] == ["mouth"]


def test_pacifier_allowed_when_nose_visible():
    r = run(FACE, NOSE, PACIFIER)
    assert r.extra["assessment"] == "clear" and r.extra["pacifier_detected"]


def test_pacifier_never_exempts_missing_nose():
    assert run(FACE, PACIFIER).extra["assessment"] == "unknown"


@pytest.mark.parametrize("label", ["blanket_on_nose", "blanket_on_mouth", "covered_face"])
def test_cover_evidence_wins_over_pacifier_and_visible_parts(label):
    r = run(FACE, NOSE, MOUTH, PACIFIER, item(label, [30, 30, 70, 80]))
    assert r.is_abnormal and r.extra["evaluable"]
    assert r.confidence == 0  # rule result is not a calibrated probability


def test_blanket_near_body_not_cover():
    assert run(FACE, NOSE, PACIFIER, item("blanket", [0, 91, 99, 99])).extra["assessment"] == "clear"


def test_bbox_overlap_alone_not_proof_of_cover():
    assert run(FACE, NOSE, MOUTH, item("blanket", [0, 45, 70, 99])).extra["assessment"] == "unknown"


def test_other_person_parts_cannot_clear_face():
    assert run(FACE, item("nose", [95, 35, 99, 50]), MOUTH).extra["assessment"] == "unknown"


def test_other_pacifier_cannot_exempt_missing_mouth():
    assert run(FACE, NOSE, item("pacifier_in_mouth", [91, 60, 99, 80])).extra["assessment"] == "unknown"


def test_multiple_faces_unknown():
    assert run(FACE, FACE, NOSE, MOUTH).extra["assessment"] == "unknown"


def test_no_face_unknown():
    assert run(NOSE, MOUTH).extra["assessment"] == "unknown"


def test_explicit_full_cover_without_face_detection():
    assert run(item("covered_face", [10, 10, 90, 90])).is_abnormal


def test_missing_model_unknown():
    r = FaceCoverDetector().detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert not r.extra["enabled"] and not r.extra["evaluable"]


def test_inference_failure_unknown():
    def fail(_):
        raise RuntimeError("inference failure")
    r = FaceCoverDetector(predict_fn=fail).detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert not r.extra["evaluable"]


def test_invalid_boxes_unknown():
    assert run(item("face", [float("nan"), 0, 90, 90]), NOSE, MOUTH).extra["assessment"] == "unknown"


def test_pacifier_over_nose_not_clear():
    assert run(FACE, NOSE, item("pacifier_in_mouth", [30, 40, 70, 70])).extra["assessment"] == "unknown"


def test_low_confidence_cover_does_not_trigger():
    assert not run(FACE, NOSE, item("blanket_on_nose", [30, 30, 70, 80], .1)).is_abnormal


def test_unworn_pacifier_does_not_exempt_mouth():
    assert run(FACE, NOSE, item("pacifier", [30, 57, 70, 80])).extra["assessment"] == "unknown"
