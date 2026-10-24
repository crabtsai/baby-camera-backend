from __future__ import annotations

import base64

import cv2
import numpy as np

from backend.detector.base_detector import BaseDetector
from backend.schemas import DetectionResult
from backend.utils.image_utils import to_gray


class MotionDetector(BaseDetector):
    # 簡易低動作提醒。
    # 注意：這不是呼吸偵測，也不是醫療判斷。
    # 只是用 frame difference 估算畫面變化量。
    name = "motion_detector"

    def __init__(
        self,
        diff_threshold: float = 8,
        low_ratio_threshold: float = 0.003,
        include_debug_images: bool = False,
    ):
        self.diff_threshold = diff_threshold
        self.low_ratio_threshold = low_ratio_threshold
        self.include_debug_images = include_debug_images
        self.prev_gray: np.ndarray | None = None

    def reset(self) -> None:
        self.prev_gray = None

    def detect(self, frame: np.ndarray) -> DetectionResult:
        # 先轉灰階並做模糊，降低雜訊與光線小幅變化造成的誤判。
        gray = to_gray(frame)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            # 第一張 frame 只能當作背景參考，還沒有上一張可比較。
            self.prev_gray = gray
            extra = {
                "step": "init_reference_frame",
                "motion_ratio": None,
                "diff_threshold": self.diff_threshold,
                "low_ratio_threshold": self.low_ratio_threshold,
            }
            if self.include_debug_images:
                extra["debug_images"] = {
                    "gray_blur": self._image_to_data_url(gray),
                }

            return DetectionResult(
                detector_name=self.name,
                event_type="low_motion_warning",
                is_abnormal=False,
                confidence=0.0,
                message="已建立第一張參考畫面，等待下一張 frame 比對動作",
                extra=extra,
            )

        prev_gray = self.prev_gray
        # absdiff 會算出每個像素與上一張的差異，差異越大代表動作越明顯。
        diff = cv2.absdiff(prev_gray, gray)
        self.prev_gray = gray

        # 將差異圖二值化：超過 diff_threshold 的像素視為「有變動」。
        _, binary = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        motion_pixels = int(np.count_nonzero(binary))
        total_pixels = int(binary.size)
        motion_ratio = motion_pixels / max(1, total_pixels)
        changed_pixel_percent = motion_ratio * 100

        is_low_motion = motion_ratio < self.low_ratio_threshold
        if is_low_motion:
            # 動作比例越接近 0，低動作信心值越高。
            confidence = 1.0 - min(1.0, motion_ratio / max(self.low_ratio_threshold, 1e-9))
        else:
            confidence = 0.0

        extra = {
            "step": "compare_with_previous_frame",
            "motion_pixels": motion_pixels,
            "total_pixels": total_pixels,
            "motion_ratio": motion_ratio,
            "changed_pixel_percent": changed_pixel_percent,
            "low_ratio_threshold": self.low_ratio_threshold,
            "diff_threshold": self.diff_threshold,
            "is_low_motion": is_low_motion,
            "decision": (
                "motion_ratio < low_ratio_threshold，所以判斷為低動作"
                if is_low_motion
                else "motion_ratio >= low_ratio_threshold，所以判斷為動作正常"
            ),
        }
        if self.include_debug_images:
            extra["debug_images"] = {
                "background_reference": self._image_to_data_url(prev_gray),
                "gray_blur": self._image_to_data_url(gray),
                "frame_diff": self._image_to_data_url(diff),
                "threshold_mask": self._image_to_data_url(binary),
            }

        return DetectionResult(
            detector_name=self.name,
            event_type="low_motion_warning",
            is_abnormal=is_low_motion,
            confidence=confidence,
            message="畫面動作偏低，請留意寶寶狀態" if is_low_motion else "畫面動作正常",
            extra=extra,
        )

    @staticmethod
    def _image_to_data_url(image: np.ndarray) -> str:
        # Detector Lab 專用：把中間處理影像轉成可直接顯示在網頁上的 data URL。
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return ""
        encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
