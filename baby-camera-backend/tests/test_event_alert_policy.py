import numpy as np

from backend.config import Settings
from backend.event_manager import EventManager
from backend.schemas import AlertPolicy, DetectionResult


class FakeDatabase:
    # 測試用記憶體資料庫，讓測試專注在 EventManager 規則。
    def __init__(self):
        self.events = []

    def insert_event(self, event):
        event.id = len(self.events) + 1
        self.events.append(event)
        return event


class FakeSnapshotService:
    # 測試策略時不真的寫入圖片檔案。
    def save_snapshot(self, frame, result):
        return f"snapshot-{result.event_type}.jpg"


class FakeAlertService:
    # 攔截告警發送，不真的呼叫 Firebase。
    def __init__(self):
        self.sent = []

    def send_alert(self, event, policy):
        self.sent.append((event, policy))
        return True


def make_manager():
    # 測試時把持續時間與冷卻時間設為 0，方便立即觸發事件。
    alert_service = FakeAlertService()
    manager = EventManager(
        settings=Settings(
            dark_duration_sec=0,
            dark_cooldown_sec=0,
            motion_duration_sec=0,
            motion_cooldown_sec=0,
        ),
        database=FakeDatabase(),
        snapshot_service=FakeSnapshotService(),
        alert_service=alert_service,
    )
    return manager, alert_service


def abnormal_result(event_type):
    return DetectionResult(
        detector_name="test_detector",
        event_type=event_type,
        is_abnormal=True,
        confidence=0.9,
        message=f"{event_type} message",
    )


def test_log_only_policy_records_event_without_line_alert():
    manager, alert_service = make_manager()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    assert manager.handle_result(frame, abnormal_result("low_motion_warning")) is None
    event = manager.handle_result(frame, abnormal_result("low_motion_warning"))

    assert event is not None
    assert event.extra["notify_line"] is False
    assert len(manager.database.events) == 1
    assert alert_service.sent == []


def test_alert_after_count_waits_until_threshold():
    manager, alert_service = make_manager()
    manager.alert_policies["baby_position_warning"] = AlertPolicy(
        duration_sec=0,
        cooldown_sec=0,
        level="critical",
        notify_line=True,
        alert_after_count=2,
    )
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    assert manager.handle_result(frame, abnormal_result("baby_position_warning")) is None
    first_event = manager.handle_result(frame, abnormal_result("baby_position_warning"))
    second_event = manager.handle_result(frame, abnormal_result("baby_position_warning"))

    assert first_event.extra["notify_line"] is False
    assert second_event.extra["notify_line"] is True
    assert len(alert_service.sent) == 1
    assert alert_service.sent[0][0] == second_event


def test_daily_alert_limit_still_records_later_events():
    manager, alert_service = make_manager()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    assert manager.handle_result(frame, abnormal_result("room_too_dark")) is None
    first_event = manager.handle_result(frame, abnormal_result("room_too_dark"))
    second_event = manager.handle_result(frame, abnormal_result("room_too_dark"))

    assert first_event.extra["notify_line"] is True
    assert second_event.extra["notify_line"] is False
    assert len(manager.database.events) == 2
    assert len(alert_service.sent) == 1
