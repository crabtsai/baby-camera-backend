from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from backend.detector.base_detector import BaseDetector
from backend.schemas import DetectionResult

logger = logging.getLogger(__name__)
YoloPredictFn = Callable[[np.ndarray], list[dict[str, Any]]]


class BabyDetector(BaseDetector):
    # 寶寶臉部偵測。
    # 第一版判斷邏輯：YOLO 有偵測到 baby_face / face，就視為寶寶在畫面中。
    name = "baby_detector"

    def __init__(
        self,
        model_path: str = "",
        face_labels: set[str] | None = None,
        confidence_threshold: float = 0.4,
        predict_fn: YoloPredictFn | None = None,
    ):
        self.model_path = model_path.strip()
        self.face_labels = face_labels or {"baby_face", "face"}
        self.confidence_threshold = confidence_threshold
        self.disabled_reason: str | None = None
        # 測試時可注入 predict_fn，不必真的載入 YOLO 模型。
        self.predict_fn = predict_fn or self._build_yolo_predict_fn()

    def _build_yolo_predict_fn(self) -> YoloPredictFn | None:
        # 沒有模型路徑時保持停用，避免缺模型導致後端啟動失敗。
        if not self.model_path:
            self.disabled_reason = "未設定 BABY_YOLO_MODEL_PATH"
            return None

        try:
            from ultralytics import YOLO
        except Exception:
            self.disabled_reason = "尚未安裝 ultralytics，無法載入 YOLO 模型"
            logger.exception(self.disabled_reason)
            return None

        try:
            model = YOLO(self.model_path)
        except Exception:
            self.disabled_reason = f"YOLO 模型載入失敗：{self.model_path}"
            logger.exception(self.disabled_reason)
            return None

        self.disabled_reason = None

        def predict(frame: np.ndarray) -> list[dict[str, Any]]:
            results = model.predict(frame, verbose=False)
            return self._normalize_yolo_results(results)

        return predict

    @staticmethod
    def _normalize_yolo_results(results: Any) -> list[dict[str, Any]]:
        # 將 ultralytics YOLO 結果轉成簡單 dict，後續判斷就不用依賴套件型別。
        detections: list[dict[str, Any]] = []
        for result in results or []:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                cls_value = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                label = str(names.get(cls_value, cls_value)).lower()
                xyxy = box.xyxy[0].tolist()
                detections.append(
                    {
                        "label": label,
                        "confidence": confidence,
                        "box": [float(value) for value in xyxy],
                    }
                )
        return detections

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if self.predict_fn is None:
            return DetectionResult(
                detector_name=self.name,
                event_type="baby_position_warning",
                is_abnormal=False,
                confidence=0.0,
                message=f"寶寶臉部偵測尚未啟用：{self.disabled_reason}",
                extra={
                    "enabled": False,
                    "model_path": self.model_path,
                    "face_labels": sorted(self.face_labels),
                    "disabled_reason": self.disabled_reason,
                },
            )

        detections = self.predict_fn(frame)
        visible_faces = self._filter_face_detections(detections)
        frame_shape = frame.shape[:2]

        if visible_faces:
            # 只要有任一張臉超過信心門檻，就視為寶寶臉部仍在畫面內。
            max_confidence = max(float(item.get("confidence", 0)) for item in visible_faces)
            return DetectionResult(
                detector_name=self.name,
                event_type="baby_position_warning",
                is_abnormal=False,
                confidence=max_confidence,
                message="已偵測到寶寶臉部",
                extra=self._build_extra(detections, visible_faces, frame_shape),
            )

        return DetectionResult(
            detector_name=self.name,
            event_type="baby_position_warning",
            is_abnormal=True,
            confidence=1.0,
            message="未偵測到寶寶臉部，請確認寶寶是否在畫面中",
            extra=self._build_extra(detections, visible_faces, frame_shape),
        )

    def _filter_face_detections(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # YOLO class 名稱轉小寫比對，讓 baby_face / Face 等命名都能統一處理。
        normalized_labels = {label.lower() for label in self.face_labels}
        return [
            detection
            for detection in detections
            if str(detection.get("label", "")).lower() in normalized_labels
            and float(detection.get("confidence", 0)) >= self.confidence_threshold
        ]

    def _build_extra(
        self,
        detections: list[dict[str, Any]],
        visible_faces: list[dict[str, Any]],
        frame_shape: tuple[int, int],
    ) -> dict[str, Any]:
        # extra 保留原始 YOLO 偵測結果，前端可用來畫框或排查 class 名稱。
        frame_height, frame_width = frame_shape
        return {
            "enabled": True,
            "model_path": self.model_path,
            "face_labels": sorted(self.face_labels),
            "confidence_threshold": self.confidence_threshold,
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "baby_face_count": len(visible_faces),
            "detections": detections,
        }
