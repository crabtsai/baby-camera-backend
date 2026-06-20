from __future__ import annotations

import cv2
import numpy as np


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    quality = max(1, min(int(quality), 100))
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()
