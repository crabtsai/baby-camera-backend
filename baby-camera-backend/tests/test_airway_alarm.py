from dataclasses import replace

import numpy as np

from tests.test_event_alert_policy import make_manager, abnormal_result


def test_local_alarm_requires_sustained_evidence_and_expires(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("backend.event_manager.time.monotonic", lambda: now[0])
    manager, outbound = make_manager()
    manager.alert_policies["face_cover_warning"] = replace(
        manager.alert_policies["face_cover_warning"], notify_line=False)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    result = abnormal_result("face_cover_warning")
    manager.handle_result(frame, result)
    now[0] += 4
    manager.handle_result(frame, result)
    assert manager.local_airway_alert() is None
    now[0] += 1
    event = manager.handle_result(frame, result)
    alert = manager.local_airway_alert()
    assert alert["id"] == event.id
    assert alert["camera_id"] == manager.camera_id
    assert outbound.sent == []
    alert["message"] = "changed"
    assert manager.local_airway_alert()["message"] == result.message
    now[0] += 59
    assert manager.handle_result(frame, result) is None  # cooldown
    assert manager.local_airway_alert() is not None
    now[0] += 1
    assert manager.local_airway_alert() is None


def test_unknown_breaks_alarm_timer_and_other_events_do_not_ring(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("backend.event_manager.time.monotonic", lambda: now[0])
    manager, _ = make_manager()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    result = abnormal_result("face_cover_warning")
    manager.handle_result(frame, result)
    now[0] += 4
    manager.handle_result(frame, replace(result, extra={"evaluable": False}))
    now[0] += 4
    manager.handle_result(frame, result)
    assert manager.local_airway_alert() is None
    for _ in range(2):
        manager.handle_result(frame, abnormal_result("low_motion_warning"))
    assert manager.local_airway_alert() is None


def test_camera_alarms_are_independent(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("backend.event_manager.time.monotonic", lambda: now[0])
    first, _ = make_manager()
    second, _ = make_manager()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    first.handle_result(frame, abnormal_result("face_cover_warning"))
    now[0] += 5
    first.handle_result(frame, abnormal_result("face_cover_warning"))
    assert first.local_airway_alert() is not None
    assert second.local_airway_alert() is None
