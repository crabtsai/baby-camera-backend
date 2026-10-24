from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from backend.detector.base_detector import BaseDetector
from backend.schemas import DetectionResult

logger = logging.getLogger(__name__)
PosePredictFn = Callable[[np.ndarray], list[dict[str, Any]]]


class ProneSleepDetector(BaseDetector):
    """Estimate face-down sleep risk from COCO pose keypoint visibility.

    A single RGB frame cannot prove that a baby is prone.  This detector therefore
    reports a risk only when a body is confidently visible while most facial
    keypoints are missing.  EventManager's duration threshold filters brief turns.
    """

    name = "prone_sleep_detector"
    FACE_KEYPOINTS = (0, 1, 2, 3, 4)  # nose, eyes, ears in COCO pose
    BODY_KEYPOINTS = (5, 6, 11, 12)  # shoulders and hips

    def __init__(
        self,
        model_path: str = "",
        keypoint_confidence_threshold: float = 0.35,
        min_body_keypoints: int = 3,
        max_visible_face_keypoints: int = 1,
        predict_fn: PosePredictFn | None = None,
    ):
        self.model_path = model_path.strip()
        self.keypoint_confidence_threshold = max(0.0, min(1.0, keypoint_confidence_threshold))
        self.min_body_keypoints = max(1, min(len(self.BODY_KEYPOINTS), min_body_keypoints))
        self.max_visible_face_keypoints = max(
            0, min(len(self.FACE_KEYPOINTS), max_visible_face_keypoints)
        )
        self.disabled_reason: str | None = None
        self.predict_fn = predict_fn or self._build_yolo_predict_fn()

    def _build_yolo_predict_fn(self) -> PosePredictFn | None:
        if not self.model_path:
            self.disabled_reason = "未設定趴睡偵測的 YOLO Pose 模型路徑"
            return None
        try:
            from ultralytics import YOLO
        except Exception:
            self.disabled_reason = "尚未安裝 ultralytics，無法載入 YOLO Pose 模型"
            logger.exception(self.disabled_reason)
            return None
        try:
            model = YOLO(self.model_path)
        except Exception:
            self.disabled_reason = f"YOLO Pose 模型載入失敗：{self.model_path}"
            logger.exception(self.disabled_reason)
            return None

        self.disabled_reason = None

        def predict(frame: np.ndarray) -> list[dict[str, Any]]:
            return self._normalize_yolo_results(model.predict(frame, verbose=False))

        return predict

    @staticmethod
    def _normalize_yolo_results(results: Any) -> list[dict[str, Any]]:
        people: list[dict[str, Any]] = []
        for result in results or []:
            keypoints = getattr(result, "keypoints", None)
            if keypoints is None:
                continue
            xy_data = getattr(keypoints, "xy", None)
            conf_data = getattr(keypoints, "conf", None)
            if xy_data is None:
                continue
            xy_rows = xy_data.cpu().tolist() if hasattr(xy_data, "cpu") else xy_data.tolist()
            if conf_data is None:
                conf_rows = [[1.0] * len(row) for row in xy_rows]
            else:
                conf_rows = conf_data.cpu().tolist() if hasattr(conf_data, "cpu") else conf_data.tolist()
            for xy_row, conf_row in zip(xy_rows, conf_rows):
                people.append(
                    {
                        "keypoints": [
                            {"x": float(xy[0]), "y": float(xy[1]), "confidence": float(conf)}
                            for xy, conf in zip(xy_row, conf_row)
                        ]
                    }
                )
        return people

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if self.predict_fn is None:
            return DetectionResult(
                detector_name=self.name,
                event_type="prone_sleep_warning",
                is_abnormal=False,
                confidence=0.0,
                message=f"趴睡偵測尚未啟用：{self.disabled_reason}",
                extra=self._extra([], False),
            )

        people = self.predict_fn(frame)
        assessments = [self._assess_person(person) for person in people]
        risky = [item for item in assessments if item["body_visible"] and item["face_down_risk"]]
        if risky:
            confidence = max(float(item["risk_confidence"]) for item in risky)
            return DetectionResult(
                detector_name=self.name,
                event_type="prone_sleep_warning",
                is_abnormal=True,
                confidence=confidence,
                message="寶寶身體可見但臉部關鍵點大多不可見，可能有趴睡風險",
                extra=self._extra(assessments, True),
            )

        message = "未偵測到可判讀的人體姿勢" if not assessments else "未發現明顯趴睡風險"
        return DetectionResult(
            detector_name=self.name,
            event_type="prone_sleep_warning",
            is_abnormal=False,
            confidence=0.0,
            message=message,
            extra=self._extra(assessments, True),
        )

    def _assess_person(self, person: dict[str, Any]) -> dict[str, Any]:
        keypoints = person.get("keypoints", [])

        def visible(index: int) -> bool:
            return index < len(keypoints) and float(keypoints[index].get("confidence", 0)) >= self.keypoint_confidence_threshold

        visible_face = sum(visible(index) for index in self.FACE_KEYPOINTS)
        visible_body = sum(visible(index) for index in self.BODY_KEYPOINTS)
        body_visible = visible_body >= self.min_body_keypoints
        face_down_risk = body_visible and visible_face <= self.max_visible_face_keypoints
        risk_confidence = (
            (visible_body / len(self.BODY_KEYPOINTS))
            * (1.0 - visible_face / len(self.FACE_KEYPOINTS))
            if face_down_risk
            else 0.0
        )
        return {
            "visible_face_keypoints": visible_face,
            "visible_body_keypoints": visible_body,
            "body_visible": body_visible,
            "face_down_risk": face_down_risk,
            "risk_confidence": round(risk_confidence, 4),
        }

    def _extra(self, assessments: list[dict[str, Any]], enabled: bool) -> dict[str, Any]:
        return {
            "enabled": enabled,
            "model_path": self.model_path,
            "keypoint_confidence_threshold": self.keypoint_confidence_threshold,
            "min_body_keypoints": self.min_body_keypoints,
            "max_visible_face_keypoints": self.max_visible_face_keypoints,
            "person_count": len(assessments),
            "assessments": assessments,
            "disabled_reason": self.disabled_reason,
        }
