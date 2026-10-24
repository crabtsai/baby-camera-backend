import numpy as np
import pytest

from backend.config import Settings
from backend.detector.detector_manager import DetectorManager
from backend.detector.safety_pipeline import SafetyOptions, SafetyPipeline, SceneModels, parse_region
from backend.schemas import AlertPolicy
from tests.test_event_alert_policy import make_manager


def frame():
    return np.random.default_rng(7).integers(30, 220, (240, 320, 3), dtype=np.uint8)


def scene():
    return {"pose_available": True, "objects_available": True,
            "people": [{"body": (.3, .2, .7, .8), "head": (.4, .2, .6, .4), "posture": "supine"}],
            "objects": []}


def pipeline(data=None, **options):
    data = scene() if data is None else data
    return SafetyPipeline(SafetyOptions(region=(.1, .1, .9, .9), present_frames=1, **options), lambda _: data)


def results(detector, image=None):
    return {r.event_type: r for r in detector.detect_many(frame() if image is None else image)}


@pytest.mark.parametrize("value", ["0,0,1", "0,0,2,1", "0.5,0,0.2,1", "nan,0,1,1", "0,0,inf,1"])
def test_invalid_regions_fail_explicitly(value):
    with pytest.raises(ValueError):
        parse_region(value)


def test_quality_gates_inference_and_clears_history():
    detector = pipeline()
    results(detector)
    assert detector._last_head is not None
    detector.scene_fn = lambda _: pytest.fail("bad frame must not reach models")
    out = results(detector, np.zeros((240, 320, 3), dtype=np.uint8))
    assert out["image_quality_warning"].is_abnormal
    assert all(not out[e].extra["evaluable"] for e in detector.EVENTS)
    assert detector._last_head is None


def test_missing_calibration_and_models_are_unknown():
    out = results(SafetyPipeline())
    assert not out["image_quality_warning"].is_abnormal
    assert all(out[e].extra["assessment"] == "unknown" for e in SafetyPipeline.EVENTS)
    out = results(SafetyPipeline(SafetyOptions(region=(0, 0, 1, 1))))
    assert all(out[e].extra["assessment"] == "unknown" for e in SafetyPipeline.EVENTS)


def test_all_spatial_rules_share_one_inference():
    calls = []
    detector = pipeline()
    detector.scene_fn = lambda image: calls.append(image) or scene()
    out = results(detector)
    assert len(calls) == 1
    assert len(out) == 6
    assert all(r.extra["evaluable"] and not r.is_abnormal for r in out.values())


def test_object_inside_mattress_and_head_overlap_are_separate_rules():
    data = scene()
    data["objects"] = [{"label": "blanket", "box": (.4, .6, .6, .8)}]
    detector = pipeline(data)
    out = results(detector)
    assert out["sleep_area_object"].is_abnormal
    assert not out["head_obstruction_warning"].is_abnormal
    data["objects"][0]["box"] = (.4, .2, .6, .4)
    assert results(detector)["head_obstruction_warning"].is_abnormal


def test_outside_objects_and_unknown_labels_do_not_trigger():
    data = scene()
    data["objects"] = [{"label": "blanket", "box": (0, 0, .09, .09)},
                       {"label": "person", "box": (.4, .2, .6, .4)}]
    out = results(pipeline(data))
    assert not out["sleep_area_object"].is_abnormal


def test_recent_head_can_support_obstruction_but_expires(monkeypatch):
    now = [100.]
    monkeypatch.setattr("backend.detector.safety_pipeline.time.monotonic", lambda: now[0])
    data = scene()
    detector = pipeline(data)
    results(detector)
    data["people"][0]["head"] = None
    data["objects"] = [{"label": "blanket", "box": (.4, .2, .6, .4)}]
    now[0] = 102.
    out = results(detector)["head_obstruction_warning"]
    assert out.is_abnormal and out.extra["head_source"] == "recent_estimate"
    now[0] = 104.
    assert not results(detector)["head_obstruction_warning"].extra["evaluable"]


def test_no_clear_assessment_from_remembered_head_without_current_evidence():
    data = scene()
    detector = pipeline(data)
    results(detector)
    data["people"][0]["head"] = None
    assert not results(detector)["head_obstruction_warning"].extra["evaluable"]


def test_multiple_people_disable_person_specific_rules():
    data = scene()
    data["people"].append(dict(data["people"][0]))
    out = results(pipeline(data))
    for event in ("head_obstruction_warning", "boundary_warning", "sleep_posture_warning"):
        assert not out[event].extra["evaluable"]


def test_boundary_and_posture_conditions():
    data = scene()
    data["people"][0].update(body=(.11, .2, .7, .8), posture="side")
    out = results(pipeline(data))
    assert out["boundary_warning"].is_abnormal
    assert out["sleep_posture_warning"].is_abnormal
    data["people"][0]["posture"] = None
    assert not results(pipeline(data))["sleep_posture_warning"].extra["evaluable"]


def test_model_failure_returns_unknown_not_clear():
    detector = pipeline()
    def fail(_):
        raise RuntimeError("inference failed")
    detector.scene_fn = fail
    out = results(detector)
    assert all(out[e].extra["assessment"] == "unknown" for e in detector.EVENTS)


def test_missing_weights_do_not_download(tmp_path):
    models = SceneModels(str(tmp_path / "missing.pt"), "", "", SafetyOptions())
    assert not models.capabilities["localization"]["available"]
    assert not models.capabilities["objects"]["available"]


@pytest.mark.parametrize("task,names", [("pose", {0: "blanket"}), ("detect", {0: "person"})])
def test_incompatible_models_report_unavailable(tmp_path, monkeypatch, task, names):
    import sys
    from types import SimpleNamespace
    weights = tmp_path / "objects.pt"
    weights.write_bytes(b"fake")
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda _: SimpleNamespace(task=task, names=names)))
    models = SceneModels("", str(weights), "", SafetyOptions())
    assert not models.capabilities["objects"]["available"]
    assert models.capabilities["objects"]["reason"]


def test_workers_and_sync_preserve_all_events_and_metadata():
    manager = DetectorManager([pipeline()])
    assert len(manager.run_all(frame())) == 6
    received = []
    manager.start_workers(lambda _, r, meta: received.append((r, meta)), max_queue_size=1)
    try:
        manager.submit_frame(frame(), {"camera_id": "test"})
        assert manager.wait_until_idle(2)
        assert len(received) == 6
        assert all(meta["camera_id"] == "test" for _, meta in received)
        assert manager.worker_status()[0]["frames_processed"] == 1
    finally:
        manager.stop_workers()


def test_unknown_and_observation_gaps_reset_event_timer(monkeypatch):
    now = [100.]
    monkeypatch.setattr("backend.event_manager.time.monotonic", lambda: now[0])
    manager, alerts = make_manager()
    event = "boundary_warning"
    manager.alert_policies[event] = AlertPolicy(duration_sec=5, cooldown_sec=0, notify_line=False)
    abnormal = SafetyPipeline()._result(event, True)
    assert manager.handle_result(frame(), abnormal) is None
    now[0] = 110.
    assert manager.handle_result(frame(), abnormal) is None  # gap resets instead of firing
    now[0] = 112.
    assert manager.handle_result(frame(), SafetyPipeline()._result(event, evaluable=False)) is None
    now[0] = 113.
    assert manager.handle_result(frame(), abnormal) is None
    now[0] = 117.
    assert manager.handle_result(frame(), abnormal) is None
    now[0] = 118.
    assert manager.handle_result(frame(), abnormal) is not None
    assert not alerts.sent


def test_camera_settings_override_shared_region(monkeypatch):
    from backend import config
    import configparser
    parser = configparser.ConfigParser()
    parser.read_string("[detectors.safety]\nsleep_region=0.1,0.1,0.9,0.9\n"
                       "[camera.test.detectors.safety]\nsleep_region=0.2,0.2,0.8,0.8\n")
    monkeypatch.setattr(config, "_CONFIG", parser)
    assert SafetyPipeline.from_settings(Settings(), "test").options.region == (.2, .2, .8, .8)


def test_lab_flattens_multi_results_and_counts_unknown():
    import base64
    import cv2
    from backend.api.detector_lab_api import DetectorLabFrame, _run_sync_lab
    _, encoded = cv2.imencode(".png", frame())
    data = _run_sync_lab([SafetyPipeline()], [DetectorLabFrame(image_data=base64.b64encode(encoded).decode())])
    assert len(data["frames"][0]["detections"]) == 6
    assert data["summary"]["safety_pipeline:boundary_warning"]["unknown"] == 1


def test_camera_pipelines_do_not_share_head_history():
    first, second = pipeline(), pipeline()
    results(first)
    assert first._last_head is not None
    assert second._last_head is None
