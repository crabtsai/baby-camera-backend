"""Opt-in audio notifications; network requests never block audio inference."""
import logging
import queue
import threading

from backend.schemas import AlertPolicy, DetectionResult

log = logging.getLogger(__name__)


class AudioNotifications:
    def __init__(self, camera_manager, snapshots, alerts):
        self.cameras, self.snapshots, self.alerts = camera_manager, snapshots, alerts
        self.queue = queue.Queue(maxsize=8)
        self.stop_event = threading.Event()
        self.thread = None
        self.last_result = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True, name='audio-notifications')
        self.thread.start()

    def prepare(self, record):
        try:
            runtime = self.cameras.get(record.camera_id)
            frame = runtime.camera.get_latest_frame() if runtime else None
            if frame is not None:
                result = DetectionResult('audio_classifier', record.event_type, True,
                                         record.confidence, record.message)
                record.snapshot_path = self.snapshots.save_snapshot(frame, result, record.camera_id)
        except Exception:
            log.exception('Audio snapshot unavailable')
        record.extra['snapshot_available'] = bool(record.snapshot_path)
        if not record.snapshot_path:
            record.message += '（目前無可用的新畫面，未附圖）'

    def submit(self, record):
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.last_result = {'event_id': record.id, 'status': 'queue_full'}
            log.error('Audio notification queue full')

    def _send(self, record):
        try:
            sent = self.alerts.send_alert(record, AlertPolicy(5, 60, notify_line=True))
            # HTTP acceptance does not prove LINE delivery (Firebase sends asynchronously).
            status = 'gateway_accepted' if sent else 'failed'
        except Exception:
            log.exception('Audio notification failed')
            status = 'failed'
        self.last_result = {'event_id': record.id, 'camera_id': record.camera_id, 'status': status}

    def _run(self):
        while not self.stop_event.is_set():
            try:
                record = self.queue.get(timeout=.2)
            except queue.Empty:
                continue
            self._send(record)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
