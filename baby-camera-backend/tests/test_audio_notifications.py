from types import SimpleNamespace
from unittest.mock import Mock
import time
import pytest

from backend.audio_notifications import AudioNotifications
from backend.schemas import EventRecord


def setup_notification(frame):
    cameras = Mock()
    cameras.get.return_value = SimpleNamespace(camera=SimpleNamespace(get_latest_frame=lambda: frame))
    snapshots = Mock()
    snapshots.save_snapshot.return_value = 'test.jpg'
    alerts = Mock()
    alerts.send_alert.return_value = True
    return AudioNotifications(cameras, snapshots, alerts)


def record():
    return EventRecord('cam-2', 'cry_warning', 'cry', .8, 'audio_classifier', id=123)


def test_snapshot_from_triggering_camera_and_queued_send():
    frame = object()
    notifier = setup_notification(frame)
    event = record()
    notifier.prepare(event)
    notifier.cameras.get.assert_called_once_with('cam-2')
    assert notifier.snapshots.save_snapshot.call_args.args[0] is frame
    assert notifier.snapshots.save_snapshot.call_args.args[2] == 'cam-2'
    assert event.snapshot_path == 'test.jpg'
    notifier.submit(event)
    notifier.alerts.send_alert.assert_not_called()
    notifier._send(notifier.queue.get_nowait())
    assert notifier.last_result['status'] == 'gateway_accepted'
    assert notifier.alerts.send_alert.call_args.args[1].notify_line


def test_stale_frame_sends_text_without_image():
    notifier = setup_notification(None)
    event = record()
    notifier.prepare(event)
    notifier.snapshots.save_snapshot.assert_not_called()
    assert event.snapshot_path is None
    assert '未附圖' in event.message
    notifier._send(event)
    notifier.alerts.send_alert.assert_called_once()


def test_snapshot_failure_keeps_text_alert():
    notifier = setup_notification(object())
    notifier.snapshots.save_snapshot.side_effect = OSError('disk full')
    event = record()
    notifier.prepare(event)
    assert not event.extra['snapshot_available']
    assert '未附圖' in event.message


def test_delivery_failure_and_queue_overflow_are_visible():
    notifier = setup_notification(None)
    notifier.alerts.send_alert.side_effect = RuntimeError('network failure')
    notifier._send(record())
    assert notifier.last_result['status'] == 'failed'
    for _ in range(9):
        notifier.submit(record())
    assert notifier.queue.qsize() == 8
    assert notifier.last_result['status'] == 'queue_full'


@pytest.mark.parametrize('enabled', [False, True])
def test_audio_event_keeps_browser_record_and_opt_in_notifications(monkeypatch, enabled):
    from backend.audio_monitor import AudioMonitor
    database = Mock()
    database.insert_event.side_effect = lambda event: event
    notifier = Mock() if enabled else None
    monitor = AudioMonitor(SimpleNamespace(cameras=[SimpleNamespace(id='cam-2')]), database, notifier)
    monitor.ffmpeg = __file__
    monkeypatch.setattr(monitor, '_capture', lambda camera: None)

    def classify(samples):
        monitor.stop_event.set()
        return {'Baby cry, infant cry': .9, 'Screaming': .1}

    monkeypatch.setattr('backend.audio_monitor.AudioClassifier', lambda path: classify)
    monitor.queues['cam-2'].put((time.monotonic(), []))
    monitor._infer()
    event = database.insert_event.call_args.args[0]
    assert event.extra['notify_line'] is enabled
    assert monitor.last_alert['camera_id'] == 'cam-2'
    if enabled:
        notifier.prepare.assert_called_once_with(event)
        notifier.submit.assert_called_once_with(event)
