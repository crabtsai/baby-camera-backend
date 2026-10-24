from __future__ import annotations

import logging
import threading
import time
from typing import Any

import cv2
import numpy as np

from backend.config import mask_camera_source

logger = logging.getLogger(__name__)


class CameraStream:
    # OpenCV 攝影機串流包裝，支援 USB Camera、RTSP、MJPEG、影片檔。
    def __init__(self, source: str | int):
        self.source = self._normalize_source(source)
        self.cap: cv2.VideoCapture | None = None
        self.lock = threading.Lock()
        self.capture_lock = threading.RLock()
        self.latest_frame: np.ndarray | None = None
        self.last_read_ok = False
        self.last_error: str | None = None
        self.last_frame_at: float | None = None
        self.last_frame_monotonic: float | None = None
        self.stale_after_sec = 5.0
        self.open_timeout_ms = 5000
        self.read_timeout_ms = 5000
        self.frames_read = 0
        self.read_failure_threshold = 3
        self.consecutive_read_failures = 0
        self.reconnect_attempts = 0
        self.reconnect_initial_delay_sec = 1.0
        self.reconnect_max_delay_sec = 30.0
        self.reconnect_delay_sec = self.reconnect_initial_delay_sec
        self.next_open_attempt_at = 0.0
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
        with self.capture_lock:
            now = time.monotonic()
            if now < self.next_open_attempt_at:
                return

            # 每次重新開啟前先釋放舊連線，避免攝影機或串流來源被重複占用。
            self.last_open_attempt_at = now
            logger.info("Opening camera source: %s", mask_camera_source(self.source))
            self._release_capture()
            # RTSP 走 FFmpeg backend 通常較穩；USB camera 則維持 OpenCV 預設 backend。
            if isinstance(self.source, str) and self.source.lower().startswith("rtsp://"):
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms,
                ])
            else:
                self.cap = cv2.VideoCapture(self.source)

            if not self.cap.isOpened():
                self.last_error = f"Camera source is not opened: {mask_camera_source(self.source)}"
                self._schedule_reconnect(self.last_error)
                self._release_capture()
                return

            # 降低串流緩衝，讓網頁畫面盡量接近即時。部分 backend 不支援時會安靜忽略。
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.consecutive_read_failures = 0
            self.next_open_attempt_at = 0.0
            self.last_error = None
            logger.info("Camera opened")

    def is_opened(self) -> bool:
        return bool(self.cap and self.cap.isOpened())

    def get_frame(self) -> np.ndarray | None:
        with self.capture_lock:
            # 若攝影機尚未開啟，等到退避時間結束後再建立連線。
            if self.cap is None or not self.cap.isOpened():
                if time.monotonic() < self.next_open_attempt_at:
                    self.last_read_ok = False
                    return None
                self.open()
                if self.cap is None or not self.cap.isOpened():
                    self.last_read_ok = False
                    return None

            try:
                ok, frame = self.cap.read()
            except cv2.error:
                ok, frame = False, None
            self.last_read_ok = bool(ok)
            if not ok or frame is None or frame.size == 0:
                self.consecutive_read_failures += 1
                self.last_error = "Failed to read frame from camera"

                # 偶發單張失敗可能只是網路抖動；連續失敗才釋放舊連線並啟動重連。
                if self.consecutive_read_failures >= self.read_failure_threshold:
                    reason = (
                        f"Camera frame read failed {self.consecutive_read_failures} times"
                    )
                    self._release_capture()
                    self._schedule_reconnect(reason)
                self.last_read_ok = False
                return None

            self.consecutive_read_failures = 0
            self.reconnect_delay_sec = self.reconnect_initial_delay_sec

            # 與 release 使用相同鎖順序，避免釋放後又發布舊幀。
            with self.lock:
                self.latest_frame = frame.copy()
                self.last_frame_at = time.time()
                self.last_frame_monotonic = time.monotonic()
                self.frames_read += 1
                self.last_error = None
        return frame

    def get_latest_frame(self) -> np.ndarray | None:
        # 回傳 copy，避免外部處理影像時改到共享的 latest_frame。
        with self.lock:
            if self.latest_frame is None or not self._frame_is_fresh():
                return None
            return self.latest_frame.copy()

    def release(self) -> None:
        with self.capture_lock:
            self._release_capture()

    def _release_capture(self) -> None:
        self.last_read_ok = False
        with self.lock:
            self.latest_frame = None
        if self.cap is not None:
            logger.info("Releasing camera")
            self.cap.release()
            self.cap = None

    def _schedule_reconnect(self, reason: str) -> None:
        self.last_error = reason
        delay = self.reconnect_delay_sec
        self.reconnect_attempts += 1
        self.next_open_attempt_at = time.monotonic() + delay
        logger.warning(
            "%s; reconnect attempt %s in %.1f seconds",
            reason,
            self.reconnect_attempts,
            delay,
        )
        self.reconnect_delay_sec = min(
            self.reconnect_max_delay_sec,
            max(self.reconnect_initial_delay_sec, delay * 2),
        )

    def _frame_is_fresh(self) -> bool:
        return (self.last_frame_monotonic is not None
                and time.monotonic() - self.last_frame_monotonic <= self.stale_after_sec)

    def status(self) -> dict[str, Any]:
        # 提供 /health 和 /camera 頁面顯示攝影機目前狀態。
        return {
            "source": mask_camera_source(self.source),
            "opened": self.is_opened(),
            "last_read_ok": self.last_read_ok,
            "has_latest_frame": self.latest_frame is not None,
            "frame_fresh": self.latest_frame is not None and self._frame_is_fresh(),
            "frame_age_sec": (round(time.monotonic() - self.last_frame_monotonic, 3)
                              if self.last_frame_monotonic is not None else None),
            "last_frame_at": self.last_frame_at,
            "frames_read": self.frames_read,
            "consecutive_read_failures": self.consecutive_read_failures,
            "reconnect_attempts": self.reconnect_attempts,
            "reconnect_delay_sec": self.reconnect_delay_sec,
            "next_reconnect_in_sec": max(
                0.0,
                round(self.next_open_attempt_at - time.monotonic(), 2),
            ),
            "last_error": self.last_error,
        }
