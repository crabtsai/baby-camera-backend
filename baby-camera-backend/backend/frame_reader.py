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
    # WebCam 讀到的畫面不會每一張都拿去做 AI 偵測。
    # 這裡負責「固定時間抽一張 frame」，再交給 callback 執行 detector。
    # 實際偵測頻率由 config.ini 的 [stream] frame_interval_sec 控制：
    #   1.0 = 約每秒偵測 1 張
    #   0.5 = 約每秒偵測 2 張
    #   2.0 = 約每 2 秒偵測 1 張
    # 注意：/stream 網頁即時畫面可以刷新得更快，但 AI 偵測只照這個間隔跑。

    def __init__(self, camera: CameraStream, interval_sec: float = 1.0):
        self.camera = camera
        # 避免設定太小造成 CPU / YOLO 負載過高，最低限制為 0.1 秒。
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
        next_sample_at = 0.0
        while not self._stop_event.is_set():
            try:
                frame = self.camera.get_frame()
            except Exception:
                logger.exception("Camera read failed")
                self._stop_event.wait(0.2)
                continue
            now = time.monotonic()
            if frame is not None and now >= next_sample_at:
                next_sample_at = now + self.interval_sec
                # 每個 callback 代表一組後續處理，例如：
                # 1. DetectorManager 執行暗光、動作、寶寶臉部等偵測
                # 2. EventManager 判斷是否記錄事件或發送 LINE 告警
                for callback in self.callbacks:
                    try:
                        callback(frame)
                    except Exception:
                        logger.exception("Frame callback failed")

            # 持續解碼以排空 RTSP 緩衝，只限制 detector 抽樣頻率。
            # 失敗時可中斷地等待，避免重連退避期間 busy loop。
            self._stop_event.wait(0.1 if frame is None else 0.001)
