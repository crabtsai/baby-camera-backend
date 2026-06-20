from __future__ import annotations

import logging
import threading
import time
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraStream:
    # OpenCV 攝影機串流包裝，支援 USB Camera、RTSP、MJPEG、影片檔。
    def __init__(self, source: str | int):
        self.source = self._normalize_source(source)
        self.cap: cv2.VideoCapture | None = None
        self.lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self.last_read_ok = False
        self.last_error: str | None = None
        self.last_frame_at: float | None = None
        self.frames_read = 0
        self.open_retry_interval_sec = 2.0
        self.last_open_attempt_at = 0.0

    @staticmethod
    def _normalize_source(source: str | int) -> str | int:
        # .env 讀到的 "0" 要轉成整數 0，OpenCV 才會當成本機攝影機索引。
        if isinstance(source, int):
            return source
        text = str(source).strip()
        if text.isdigit():
            return int(text)
        return text

    def open(self) -> None:
        # 每次重新開啟前先釋放舊連線，避免攝影機或串流來源被重複占用。
        self.last_open_attempt_at = time.monotonic()
        logger.info("Opening camera source: %s", self.source)
        self.release()
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            self.last_error = f"Camera source is not opened: {self.source}"
            logger.warning(self.last_error)
            return

        self.last_error = None
        logger.info("Camera opened")

    def is_opened(self) -> bool:
        return bool(self.cap and self.cap.isOpened())

    def get_frame(self) -> np.ndarray | None:
        # 若攝影機尚未開啟，先嘗試建立連線。
        if self.cap is None or not self.cap.isOpened():
            if time.monotonic() - self.last_open_attempt_at < self.open_retry_interval_sec:
                self.last_read_ok = False
                return None
            self.open()
            time.sleep(0.2)
            if self.cap is None or not self.cap.isOpened():
                self.last_read_ok = False
                return None

        ok, frame = self.cap.read()
        self.last_read_ok = bool(ok)
        if not ok or frame is None:
            self.last_error = "Failed to read frame from camera"
            logger.warning(self.last_error)
            return None

        # 儲存最新 frame，讓 /stream 可以直接取用，不必每個連線都重新讀攝影機。
        with self.lock:
            self.latest_frame = frame.copy()
            self.last_frame_at = time.time()
            self.frames_read += 1
            self.last_error = None
        return frame

    def get_latest_frame(self) -> np.ndarray | None:
        # 回傳 copy，避免外部處理影像時改到共享的 latest_frame。
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def release(self) -> None:
        if self.cap is not None:
            logger.info("Releasing camera")
            self.cap.release()
            self.cap = None

    def status(self) -> dict[str, Any]:
        # 提供 /health 和 /camera 頁面顯示攝影機目前狀態。
        return {
            "source": str(self.source),
            "opened": self.is_opened(),
            "last_read_ok": self.last_read_ok,
            "has_latest_frame": self.latest_frame is not None,
            "last_frame_at": self.last_frame_at,
            "frames_read": self.frames_read,
            "last_error": self.last_error,
        }
