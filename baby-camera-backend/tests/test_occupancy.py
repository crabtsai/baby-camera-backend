import numpy as np

from backend.detector.occupancy import OccupancyTracker, summarize_occupancy
from backend.detector.safety_pipeline import SafetyOptions, SafetyPipeline
from backend.detector.motion_detector import MotionDetector
from backend.detector.base_detector import BaseDetector
from backend.schemas import DetectionResult
from tests.test_safety_pipeline import frame, scene, results


def test_presence_requires_consecutive_observations_and_resets_after_gap(monkeypatch):
    clock = [100.]
    monkeypatch.setattr("backend.detector.occupancy.time.monotonic", lambda: clock[0])
    tracker = OccupancyTracker()
    assert tracker.observe(1, "frame")["state"] == "unknown"
    assert tracker.observe(1, "frame")["state"] == "unknown"
    assert tracker.observe(1, "frame")["state"] == "present"
    assert tracker.observe(None, "frame")["state"] == "unknown"
    assert tracker.observe(1, "frame")["consecutive_frames"] == 1
    clock[0] += 6
    assert tracker.status()["state"] == "unknown"
    assert tracker.observe(1, "frame")["consecutive_frames"] == 1


def test_empty_evidence_is_not_identity_verification():
    tracker = OccupancyTracker()
    for _ in range(4):
        assert tracker.observe(0, "sleep_region")["state"] == "unknown"
    out = tracker.observe(0, "sleep_region")
    assert out["state"] == "absent"
    assert not out["identity_verified"]
    assert tracker.observe(2, "sleep_region")["state"] == "unknown"


def camera(camera_id, tracker, fresh=True):
    return {"camera_id": camera_id, "camera": {"frame_fresh": fresh}, "occupancy": tracker.status()}


def test_baby_can_move_between_beds_without_reconfiguring():
    a, b = OccupancyTracker(), OccupancyTracker()
    for _ in range(5):
        a.observe(1, "frame")
        b.observe(0, "frame")
    assert summarize_occupancy([camera("a", a), camera("b", b)])["suggested_camera_id"] == "a"
    for _ in range(3):
        a.observe(0, "frame")
        b.observe(1, "frame")
    assert summarize_occupancy([camera("a", a), camera("b", b)])["suggested_camera_id"] is None
    for _ in range(2):
        a.observe(0, "frame")
        b.observe(1, "frame")
    assert summarize_occupancy([camera("a", a), camera("b", b)])["suggested_camera_id"] == "b"


def test_multiple_candidates_or_disconnected_camera_prevent_auto_selection():
    a, b = OccupancyTracker(1), OccupancyTracker(1)
    a.observe(1, "frame")
    b.observe(1, "frame")
    out = summarize_occupancy([camera("a", a), camera("b", b)])
    assert out["selection"] == "multiple_candidates"
    assert out["suggested_camera_id"] is None
    out = summarize_occupancy([camera("a", a), camera("b", b, False)])
    assert out["selection"] == "uncertain"
    assert out["suggested_camera_id"] is None


def test_presence_runs_without_mattress_calibration():
    detector = SafetyPipeline(SafetyOptions(present_frames=1), lambda _: scene())
    out = results(detector)
    assert out["bed_occupancy"].extra["state"] == "present"
    assert out["bed_occupancy"].extra["scope"] == "frame"
    assert not out["boundary_warning"].extra["evaluable"]


class SubjectDetector(BaseDetector):
    name = "baby_detector"

    def __init__(self):
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return DetectionResult(self.name, "baby_position_warning", True, 1., "face not found")


def test_same_frame_presence_gates_old_missing_face_and_motion_rules():
    data = scene()
    data["people"] = []
    detector = SafetyPipeline(SafetyOptions(present_frames=2, absent_frames=2), lambda _: data)
    face, motion = SubjectDetector(), MotionDetector()
    detector.attach_dependents([face, motion])
    results(detector)
    out = results(detector)
    assert out["bed_occupancy"].extra["state"] == "absent"
    assert not out["baby_position_warning"].extra["evaluable"]
    assert face.calls == 0 and motion.prev_gray is None
    data["people"] = scene()["people"]
    results(detector)
    assert face.calls == 0
    out = results(detector)
    assert out["baby_position_warning"].is_abnormal
    assert face.calls == 1 and motion.prev_gray is not None
    # One lost observation immediately stops subject warnings; it does not yet declare empty.
    data["people"] = []
    out = results(detector)
    assert out["bed_occupancy"].extra["state"] == "unknown"
    assert not out["baby_position_warning"].is_abnormal
    assert motion.prev_gray is None


def test_bad_quality_and_model_failure_never_count_as_absence():
    detector = SafetyPipeline(SafetyOptions(present_frames=1), lambda _: scene())
    assert results(detector)["bed_occupancy"].extra["state"] == "present"
    for _ in range(6):
        out = results(detector, np.zeros((240, 320, 3), np.uint8))
        assert out["bed_occupancy"].extra["state"] == "unknown"
    detector.scene_fn = lambda _: {"pose_available": False}
    for _ in range(6):
        assert results(detector)["bed_occupancy"].extra["state"] == "unknown"


def test_empty_bed_still_checks_objects():
    data = scene()
    data["people"] = []
    data["objects"] = [{"label": "pillow", "box": (.4, .4, .6, .6)}]
    detector = SafetyPipeline(SafetyOptions(region=(0, 0, 1, 1), absent_frames=1), lambda _: data)
    out = results(detector)
    assert out["bed_occupancy"].extra["state"] == "absent"
    assert out["sleep_area_object"].is_abnormal
    assert not out["sleep_posture_warning"].extra["evaluable"]


def test_dependent_error_is_visible_and_does_not_break_presence():
    class Broken(SubjectDetector):
        def detect(self, image):
            raise RuntimeError("broken model")
    detector = SafetyPipeline(SafetyOptions(present_frames=1), lambda _: scene())
    detector.attach_dependents([Broken()])
    out = results(detector)
    assert out["bed_occupancy"].extra["state"] == "present"
    assert not out["baby_position_warning"].extra["evaluable"]
    assert not detector.operational
    assert detector.capabilities["dependent_detectors"]["baby_detector"]["last_error"] == "broken model"


def test_runtime_builder_uses_ordered_gate_instead_of_independent_subject_workers(monkeypatch):
    from backend.config import Settings
    from backend.detector.detector_manager import DetectorManager
    data = scene()
    detector = SafetyPipeline(SafetyOptions(present_frames=1), lambda _: data)
    face = SubjectDetector()
    monkeypatch.setattr(SafetyPipeline, "from_settings", lambda *_: detector)
    monkeypatch.setattr("backend.detector.detector_manager.BabyDetector", lambda **_: face)
    manager = DetectorManager.from_settings(Settings(enable_safety_detector=True,
        enable_baby_detector=True, enable_motion_detector=True, enable_dark_detector=False,
        enable_face_cover_detector=False, enable_face_parts_detector=False, enable_prone_sleep_detector=False))
    assert manager.detectors == [detector]
    assert {d.name for d in detector.dependents} == {"baby_detector", "motion_detector"}
    manager.run_all(frame())
    assert face.calls == 1
    assert manager.occupancy_status()["state"] == "present"
    data["people"] = []
    output = manager.run_all(frame())
    assert face.calls == 1
    assert all(not r.is_abnormal for r in output if r.detector_name == "baby_detector")
