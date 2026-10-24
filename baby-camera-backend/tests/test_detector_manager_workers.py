import threading

import numpy as np

from backend.detector.base_detector import BaseDetector
from backend.detector.detector_manager import DetectorManager
from backend.config import Settings
from backend.schemas import DetectionResult


class FakeDetector(BaseDetector):
    # 測試用 detector：不做影像處理，只回傳自己的名稱。
    def __init__(self, name):
        self.name = name

    def detect(self, frame):
        return DetectionResult(
            detector_name=self.name,
            event_type=f"{self.name}_event",
            is_abnormal=False,
            confidence=0.0,
            message=f"{self.name} ok",
        )


def test_detector_workers_receive_their_own_frame():
    # WebCam 即時模式會把同一張 frame 廣播給每個 detector worker。
    manager = DetectorManager([FakeDetector("a"), FakeDetector("b")])
    received = []
    lock = threading.Lock()
    done = threading.Event()

    def callback(frame, result, metadata):
        with lock:
            received.append(result.detector_name)
            if len(received) == 2:
                done.set()

    manager.start_workers(callback)
    try:
        manager.submit_frame(np.zeros((8, 8, 3), dtype=np.uint8))
        assert done.wait(timeout=2)
    finally:
        manager.stop_workers()

    assert sorted(received) == ["a", "b"]


def test_detector_worker_queue_keeps_latest_ten_frames():
    # detector 還沒開始處理時，queue 最多保留 10 張；第 11 張後會丟最舊 frame。
    received = []
    worker = DetectorManager([FakeDetector("slow")])

    def callback(frame, result, metadata):
        received.append(result.detector_name)

    worker.start_workers(callback)
    try:
        detector_worker = worker.workers[0]
        detector_worker.stop_event.set()
        detector_worker.thread.join(timeout=2)

        for index in range(12):
            frame = np.full((8, 8, 3), index, dtype=np.uint8)
            detector_worker.submit_frame(frame)

        status = detector_worker.status()
        assert status["max_queue_size"] == 10
        assert status["queued_frames"] == 10
        assert status["frames_dropped"] == 2
    finally:
        worker.stop_workers()


def test_manager_builds_enabled_prone_sleep_detector():
    settings = Settings(
        enable_safety_detector=False,
        enable_dark_detector=False,
        enable_motion_detector=False,
        enable_baby_detector=False,
        enable_face_cover_detector=False,
        enable_prone_sleep_detector=True,
        prone_sleep_yolo_model_path="",
    )

    manager = DetectorManager.from_settings(settings)

    assert manager.names() == ["prone_sleep_detector"]


def test_failed_detector_does_not_block_idle_wait():
    class FailingDetector(FakeDetector):
        def detect(self, frame):
            raise RuntimeError("test inference failure")

    manager = DetectorManager([FailingDetector("failure")])
    manager.start_workers(lambda *_: None, max_queue_size=1)
    try:
        manager.submit_frame(np.zeros((2, 2, 3), dtype=np.uint8))
        assert manager.wait_until_idle(timeout_sec=2)
        status = manager.worker_status()[0]
        assert status["frames_failed"] == 1
        assert status["frames_processed"] == 0
        assert status["last_error"] == "test inference failure"
    finally:
        manager.stop_workers()
