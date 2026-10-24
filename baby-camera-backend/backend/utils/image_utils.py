from __future__ import annotations

import cv2
import numpy as np


def to_gray(frame: np.ndarray) -> np.ndarray:
    # OpenCV 讀到的彩色影像是 BGR；多數基礎影像處理先轉灰階較好算。
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    # MJPEG 串流需要每張 frame 都壓成 JPEG bytes。
    # quality 限制在 1~100，避免設定錯誤造成 OpenCV 例外或檔案過大。
    quality = max(1, min(int(quality), 100))
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()
