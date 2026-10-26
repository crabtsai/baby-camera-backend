from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.config import Settings
from backend.detector.baby_face_parts import BabyFacePartsDetector

FRAME = np.zeros((100, 200, 3), dtype=np.uint8)


def mesh():
    return np.random.default_rng(12).uniform(.2, .8, (478, 3)).tolist()


def test_localization_never_claims_visibility_or_safety():
    instance = BabyFacePartsDetector(predict_fn=lambda _: [mesh()])
    result = instance.detect(FRAME)
    assert result.extra["localization_status"] == "located"
    assert result.extra["assessment"] == "unknown"
    assert not result.extra["evaluable"] and not result.is_abnormal
    assert not result.extra["visibility_confirmed"]
    assert not result.extra["obstruction_detection_supported"]
    assert result.extra["offline"]
    assert len(result.extra["detections"]) == 3
    nose = result.extra["detections"][1]
    points = np.array(mesh())[[1, 2, 98, 327, 168], :2] * [200, 100]
    assert nose["box"] == [*points.min(axis=0), *points.max(axis=0)]
    assert nose["estimated"] and nose["confidence"] == 0


@pytest.mark.parametrize("faces", [[], [mesh(), mesh()], [mesh()[:200]], [[[float('nan')]*3]*478], [[[.5]*3]*478]])
def test_invalid_empty_or_multiple_faces_do_not_localize(faces):
    result = BabyFacePartsDetector(predict_fn=lambda _: faces).detect(FRAME)
    assert result.extra["localization_status"] == "unknown"
    assert not result.extra["detections"] and not result.is_abnormal


def test_out_of_frame_mesh_is_unknown():
    points = mesh()
    points[1][0] = -0.1
    result = BabyFacePartsDetector(predict_fn=lambda _: [points]).detect(FRAME)
    assert result.extra["localization_status"] == "unknown"


def test_failure_clears_preceding_success():
    instance = BabyFacePartsDetector(predict_fn=lambda _: [mesh()])
    instance.detect(FRAME)
    def fail(_):
        raise RuntimeError("internal-details")
    instance.predict_fn = fail
    result = instance.detect(FRAME)
    assert not instance.operational
    assert result.extra["detections"] == []
    assert "internal-details" not in result.message
    assert instance.last_results[0]["localization_status"] == "unknown"


def test_missing_model_no_network(tmp_path, monkeypatch):
    import socket
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("unexpected network"))
    instance = BabyFacePartsDetector(model_path=tmp_path / "missing.task")
    assert instance.predict_fn is None
    assert instance.detect(FRAME).extra["localization_status"] == "unknown"


def test_close_releases_native_resources():
    instance = BabyFacePartsDetector(predict_fn=lambda _: [])
    calls = []
    instance._landmarker = SimpleNamespace(close=lambda: calls.append(True))
    instance.close()
    instance.close()
    assert calls == [True]


def test_pipeline_still_gates_bad_frames():
    from tests.test_safety_pipeline import pipeline, results
    instance = BabyFacePartsDetector(predict_fn=lambda _: [mesh()])
    safety = pipeline()
    safety.attach_dependents([instance])
    assert results(safety)["face_parts_visibility"].extra["localization_status"] == "located"
    instance.predict_fn = lambda _: pytest.fail("bad image reached model")
    assert results(safety, FRAME)["face_parts_visibility"].extra["assessment"] == "unknown"


def test_settings_have_local_model_and_no_cloud_credentials():
    settings = Settings()
    assert not hasattr(settings, "face_parts_api_key")
    assert not hasattr(settings, "face_parts_api_url")


def test_real_model_cpu_inference_offline(monkeypatch):
    if not Path("models/face_landmarker.task").is_file():
        pytest.skip("Official model is not downloaded on this host")
    pytest.importorskip("mediapipe")
    import socket
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: pytest.fail("unexpected network"))
    instance = BabyFacePartsDetector()
    try:
        result = instance.detect(FRAME)
        assert instance.operational
        assert result.extra["localization_status"] == "unknown"
    finally:
        instance.close()
