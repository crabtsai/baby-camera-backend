import time
import threading

import numpy as np

from backend.camera_stream import CameraStream
from backend.frame_reader import FrameReader
from tests.test_camera_stream_reconnect import FakeCapture


def test_expired_frame_is_not_returned():
    stream = CameraStream(0)
    stream.cap = FakeCapture([(True, np.ones((4, 4, 3), dtype=np.uint8))])
    stream.get_frame()
    assert stream.get_latest_frame() is not None
    stream.last_frame_monotonic = time.monotonic() - 10
    assert stream.get_latest_frame() is None
    assert not stream.status()["frame_fresh"]


def test_disconnect_clears_cached_frame():
    stream = CameraStream(0)
    stream.cap = FakeCapture([(True, np.ones((4, 4, 3), dtype=np.uint8))] + [(False, None)] * 3)
    stream.get_frame()
    for _ in range(3):
        stream.get_frame()
    assert stream.get_latest_frame() is None
    assert not stream.status()["last_read_ok"]


def test_empty_frame_counts_as_failure():
    stream = CameraStream(0)
    stream.cap = FakeCapture([(True, np.zeros((0, 0, 3), dtype=np.uint8))])
    assert stream.get_frame() is None
    assert stream.frames_read == 0


def test_reader_drains_frames_between_detector_samples():
    class Camera:
        count = 0

        def get_frame(self):
            self.count += 1
            time.sleep(0.005)
            return np.zeros((2, 2, 3), dtype=np.uint8)

    camera = Camera()
    reader = FrameReader(camera, interval_sec=0.1)
    samples = []
    done = threading.Event()

    def collect(frame):
        samples.append(camera.count)
        if len(samples) == 3:
            done.set()

    reader.register_callback(collect)
    reader.start()
    try:
        assert done.wait(2)
    finally:
        reader.stop()
    assert samples[1] - samples[0] > 1
    assert samples[2] - samples[1] > 1


def test_rtsp_timeouts_are_set_on_open(monkeypatch):
    import cv2
    calls = []
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: calls.append(args) or FakeCapture([]))
    stream = CameraStream("rtsp://camera/stream")
    stream.open()
    assert calls[0][2] == [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                           cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]


def test_decoder_exception_enters_reconnect_path():
    import cv2

    class BrokenCapture(FakeCapture):
        def read(self):
            raise cv2.error("decoder failed")

    stream = CameraStream(0)
    stream.cap = BrokenCapture([])
    for _ in range(3):
        assert stream.get_frame() is None
    assert stream.cap is None
    assert stream.reconnect_attempts == 1
