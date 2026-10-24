import numpy as np

from backend.camera_stream import CameraStream


class FakeCapture:
    def __init__(self, reads, opened=True):
        self.reads = list(reads)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened and not self.released

    def read(self):
        if self.reads:
            return self.reads.pop(0)
        return False, None

    def release(self):
        self.released = True

    def set(self, *_args):
        return True


def test_consecutive_read_failures_release_and_schedule_reconnect():
    stream = CameraStream("rtsp://user:secret@camera/stream2")
    capture = FakeCapture([(False, None)] * 3)
    stream.cap = capture

    assert stream.get_frame() is None
    assert stream.cap is capture
    assert stream.get_frame() is None
    assert stream.cap is capture
    assert stream.get_frame() is None

    assert capture.released is True
    assert stream.cap is None
    assert stream.reconnect_attempts == 1
    assert stream.next_open_attempt_at > 0


def test_successful_frame_clears_consecutive_failure_count():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    stream = CameraStream(0)
    stream.consecutive_read_failures = 2
    stream.cap = FakeCapture([(True, frame)])

    result = stream.get_frame()

    assert result is not None
    assert stream.consecutive_read_failures == 0
    assert stream.frames_read == 1


def test_reconnect_backoff_doubles_until_maximum():
    stream = CameraStream(0)

    delays = []
    for _ in range(7):
        delays.append(stream.reconnect_delay_sec)
        stream._schedule_reconnect("test")

    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


def test_open_without_valid_frames_does_not_reset_backoff(monkeypatch):
    import cv2
    stream = CameraStream("rtsp://camera/stream")
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: FakeCapture([(False, None)] * 3))
    for _ in range(3):
        stream.next_open_attempt_at = 0
        for _ in range(3):
            stream.get_frame()
    assert stream.reconnect_delay_sec == 8


def test_good_frame_after_reconnect_resets_backoff(monkeypatch):
    import cv2
    stream = CameraStream("rtsp://camera/stream")
    stream.reconnect_delay_sec = 16
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: FakeCapture([
        (True, np.ones((4, 4, 3), dtype=np.uint8))]))
    assert stream.get_frame() is not None
    assert stream.reconnect_delay_sec == 1
