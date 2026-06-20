from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import numpy as np

from backend.camera_stream import CameraStream

logger = logging.getLogger(__name__)
FrameCallback = Callable[[np.ndarray], None]


class FrameReader:
    # 固定時間抽幀，將 frame 丟給 callback。
    # 即時串流可以很快，但 AI 偵測通常 1 秒 1 張就夠。

    def __init__(self, camera: CameraStream, interval_sec: float = 1.0):
        self.camera = camera
        self.interval_sec = max(0.1, interval_sec)
        self.callbacks: list[FrameCallback] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def register_callback(self, callback: FrameCallback) -> None:
        self.callbacks.append(callback)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        logger.info("Starting frame reader")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        logger.info("Stopping frame reader")
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            frame = self.camera.get_frame()
            if frame is not None:
                for callback in self.callbacks:
                    try:
                        callback(frame)
                    except Exception:
                        logger.exception("Frame callback failed")

            elapsed = time.monotonic() - started
            sleep_sec = max(0.01, self.interval_sec - elapsed)
            time.sleep(sleep_sec)
