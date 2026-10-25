import base64

import cv2
import numpy as np
import pytest

from backend.config import Settings
from backend.detector.baby_face_parts import BabyFacePartsDetector

FRAME = np.zeros((100, 100, 3), dtype=np.uint8)


def prediction(label, x, y, width, height, confidence=.9):
    return {"class": label, "x": x, "y": y, "width": width, "height": height, "confidence": confidence}


HEAD = prediction("HEAD", 50, 50, 80, 80)
NOSE = prediction("NOSE", 50, 45, 10, 10)
MOUTH = prediction("MOUTH", 50, 70, 20, 10)


def detector(*predictions):
    return BabyFacePartsDetector(predict_fn=lambda _: BabyFacePartsDetector.normalize({"predictions": list(predictions)}))


def test_visible_parts_not_an_obstruction_event():
    result = detector(HEAD, NOSE, MOUTH).detect(FRAME)
    assert result.extra["assessment"] == "clear"
    assert not result.is_abnormal
    assert result.event_type == "face_parts_visibility"
    assert not result.extra["obstruction_detection_supported"]
    assert result.extra["detections"][0]["box"] == [10, 10, 90, 90]


@pytest.mark.parametrize("parts", [(HEAD, NOSE), (HEAD, MOUTH), (NOSE, MOUTH), (HEAD, HEAD, NOSE, MOUTH)])
def test_missing_or_ambiguous_anatomy_is_unknown(parts):
    result = detector(*parts).detect(FRAME)
    assert result.extra["assessment"] == "unknown"
    assert not result.is_abnormal


def test_other_heads_parts_and_low_confidence_do_not_clear():
    assert detector(HEAD, NOSE, prediction("MOUTH", 95, 70, 4, 4)).detect(FRAME).extra["assessment"] == "unknown"
    assert detector(HEAD, NOSE, {**MOUTH, "confidence": .1}).detect(FRAME).extra["assessment"] == "unknown"


def test_outage_clears_previous_success():
    instance = detector(HEAD, NOSE, MOUTH)
    assert instance.detect(FRAME).extra["assessment"] == "clear"
    def failure(_):
        raise ValueError("secret-key-must-not-appear")
    instance.predict_fn = failure
    result = instance.detect(FRAME)
    assert not instance.operational
    assert result.extra["assessment"] == "unknown"
    assert result.extra["detections"] == []
    assert "secret-key" not in result.message
    assert instance.last_results[0]["assessment"] == "unknown"


@pytest.mark.parametrize("url", ["https://serverless.roboflow.com", "http://example.com", "http://127.0.0.1:9001/path", "http://key@127.0.0.1:9001"])
def test_no_remote_upload_without_explicit_cloud_setting(url):
    instance = BabyFacePartsDetector(api_key="secret", api_url=url)
    assert instance.predict_fn is None


def test_missing_key_disables_without_network():
    instance = BabyFacePartsDetector()
    assert instance.predict_fn is None
    assert instance.detect(FRAME).extra["assessment"] == "unknown"


def test_http_payload_and_no_redirect_or_proxy(monkeypatch):
    instance = BabyFacePartsDetector(api_key="secret")
    class Response:
        status_code = 200
        def json(self):
            return {"predictions": [HEAD, NOSE, MOUTH]}
    def post(url, **kwargs):
        assert url == "http://127.0.0.1:9001/infer/object_detection"
        assert kwargs["headers"]["Authorization"] == "Bearer secret"
        assert kwargs["allow_redirects"] is False
        assert kwargs["json"]["disable_model_monitoring"] is True
        assert kwargs["json"]["model_id"] == "baby-face-parts/2"
        raw = base64.b64decode(kwargs["json"]["image"]["value"])
        assert cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR).shape == FRAME.shape
        return Response()
    monkeypatch.setattr(instance.session, "post", post)
    assert instance.session.trust_env is False
    assert instance.detect(FRAME).extra["assessment"] == "clear"


@pytest.mark.parametrize("payload", [{}, {"predictions": [prediction("blanket", 1, 1, 1, 1)]}, {"predictions": [{**NOSE, "x": float("nan")}]}])
def test_invalid_responses_rejected(payload):
    with pytest.raises(ValueError):
        BabyFacePartsDetector.normalize(payload)


def test_settings_hide_key_from_repr():
    assert "test-secret" not in repr(Settings(face_parts_api_key="test-secret"))


def test_pipeline_gates_inference_and_reports_connection_health():
    from tests.test_safety_pipeline import pipeline, results
    instance = detector(HEAD, NOSE, MOUTH)
    safety = pipeline()
    safety.attach_dependents([instance])
    assert results(safety)["face_parts_visibility"].extra["assessment"] == "clear"
    def failure(_):
        raise RuntimeError("offline")
    instance.predict_fn = failure
    assert results(safety)["face_parts_visibility"].extra["assessment"] == "unknown"
    assert not safety.operational
    instance.predict_fn = lambda _: pytest.fail("bad image should not reach Roboflow")
    assert results(safety, FRAME)["face_parts_visibility"].extra["assessment"] == "unknown"


def test_redirect_does_not_clear_visibility(monkeypatch):
    instance = BabyFacePartsDetector(api_key="secret")
    class Response:
        status_code = 307
    monkeypatch.setattr(instance.session, "post", lambda *a, **k: Response())
    assert instance.detect(FRAME).extra["assessment"] == "unknown"


def test_manager_uses_independent_parts_detector():
    from backend.detector.detector_manager import DetectorManager
    settings = Settings(enable_face_parts_detector=True, enable_face_cover_detector=False,
                        enable_safety_detector=False)
    manager = DetectorManager.from_settings(settings)
    assert "baby_face_parts_detector" in manager.names()
