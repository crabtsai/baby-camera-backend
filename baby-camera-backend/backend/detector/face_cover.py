from __future__ import annotations

import logging
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np

from backend.detector.base_detector import BaseDetector
from backend.schemas import DetectionResult

logger = logging.getLogger(__name__)
YoloPredictFn = Callable[[np.ndarray], list[dict[str, Any]]]


class FaceCoverDetector(BaseDetector):
    # 臉部遮蔽偵測。
    # Only explicit obstruction evidence may trigger a cover event; missing parts are unknown.
    name = "face_cover_detector"

    def __init__(
        self,
        model_path: str = "",
        face_labels: set[str] | None = None,
        nose_labels: set[str] | None = None,
        mouth_labels: set[str] | None = None,
        face_confidence_threshold: float = 0.4,
        part_confidence_threshold: float = 0.25,
        require_nose: bool = True,
        require_mouth: bool = True,
        predict_fn: YoloPredictFn | None = None,
    ):
        self.model_path = model_path.strip()
        self.face_labels = face_labels or {"face", "baby_face"}
        self.nose_labels = nose_labels or {"nose"}
        self.mouth_labels = mouth_labels or {"mouth"}
        self.face_confidence_threshold = face_confidence_threshold
        self.part_confidence_threshold = part_confidence_threshold
        # Legacy toggles remain accepted for config/Lab compatibility. Neither may
        # waive visibility in the conservative airway assessment.
        self.require_nose = require_nose
        self.require_mouth = require_mouth
        # 測試時可注入 predict_fn，不必真的載入 YOLO 模型。
        self.disabled_reason = None
        self.predict_fn = predict_fn or self._build_yolo_predict_fn()

    def _build_yolo_predict_fn(self) -> YoloPredictFn | None:
        # 沒有模型路徑時保持停用，避免目前開發階段缺模型導致後端啟動失敗。
        if not self.model_path or not Path(self.model_path).is_file():
            self.disabled_reason = "尚未提供本機口鼻／棉被／奶嘴模型"
            return None

        try:
            from ultralytics import YOLO
        except Exception:
            logger.exception("ultralytics 尚未安裝，無法載入臉部遮蔽 YOLO 模型")
            return None

        try:
            model = YOLO(self.model_path)
            required = {"face", "nose", "mouth", "pacifier_in_mouth", "blanket",
                        "blanket_on_nose", "blanket_on_mouth", "covered_face"}
            if model.task != "detect" or not required.issubset(set(model.names.values())):
                raise ValueError("模型必須為 detect 且包含：" + ", ".join(sorted(required)))
        except Exception as exc:
            self.disabled_reason = str(exc)
            logger.exception("臉部遮蔽 YOLO 模型載入失敗：%s", self.model_path)
            return None

        def predict(frame: np.ndarray) -> list[dict[str, Any]]:
            results = model.predict(frame, verbose=False, conf=min(self.face_confidence_threshold, self.part_confidence_threshold))
            return self._normalize_yolo_results(results)

        return predict

    @staticmethod
    def _normalize_yolo_results(results: Any) -> list[dict[str, Any]]:
        # 將 ultralytics YOLO 結果轉成簡單 dict，後續判斷就不用依賴特定套件型別。
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

    def _outcome(self, detections, shape, assessment, message, **details):
        faces = self._filter_by_labels(detections, self.face_labels, self.face_confidence_threshold)
        noses = self._filter_by_labels(detections, self.nose_labels, self.part_confidence_threshold)
        mouths = self._filter_by_labels(detections, self.mouth_labels, self.part_confidence_threshold)
        return DetectionResult(self.name, "face_cover_warning", assessment == "abnormal", 0., message,
            {**self._build_extra(detections, faces, noses, mouths, shape),
             "enabled": self.predict_fn is not None, "assessment": assessment,
             "evaluable": assessment != "unknown", "confidence_kind": "rule_based_not_probability",
             "pacifier_allowed": True, **details})

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if self.predict_fn is None:
            return self._outcome([], frame.shape[:2], "unknown",
                                 self.disabled_reason or "口鼻遮擋模型不可用")
        try:
            return self._decide_from_detections(self.predict_fn(frame), frame.shape[:2])
        except Exception:
            logger.exception("口鼻遮擋推論失敗")
            return self._outcome([], frame.shape[:2], "unknown", "口鼻遮擋推論失敗")

    @staticmethod
    def _box(detection):
        try:
            box = tuple(float(v) for v in detection.get("box", []))
            if len(box) == 4 and all(np.isfinite(v) for v in box) and box[2] > box[0] and box[3] > box[1]:
                return box
        except (ValueError, TypeError):
            pass
        return None

    @classmethod
    def _inside(cls, detection, face):
        box, region = cls._box(detection), cls._box(face)
        if box is None or region is None:
            return False
        # Require the entire part box inside this face, not a part from another person.
        return region[0] <= box[0] < box[2] <= region[2] and region[1] <= box[1] < box[3] <= region[3]

    @classmethod
    def _overlaps(cls, a, b):
        a, b = cls._box(a), cls._box(b)
        if a is None or b is None:
            return False
        return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])

    def _decide_from_detections(self, detections, frame_shape=None):
        def outcome(state, message, **details):
            return self._outcome(detections, frame_shape, state, message, **details)

        faces = self._filter_by_labels(detections, self.face_labels, self.face_confidence_threshold)
        evidence = self._filter_by_labels(detections,
            {"blanket_on_nose", "blanket_on_mouth", "covered_face"}, self.face_confidence_threshold)
        if len(faces) > 1:
            return outcome("unknown", "多人臉，無法確認監測對象")
        # Full covering can remove the face detection itself. The pipeline's occupancy gate still applies.
        if not faces:
            full = [d for d in evidence if d['label'] == 'covered_face' and self._box(d)]
            if len(full) == 1:
                return outcome("abnormal", "疑似臉部被覆蓋，請立即查看", obstruction=full)
            return outcome("unknown", "未偵測到臉部，無法判讀口鼻遮擋")
        face = faces[0]
        if self._box(face) is None:
            return outcome("unknown", "缺少臉部位置，無法對應口鼻")
        covering = [d for d in evidence if self._inside(d, face)]
        if covering:
            return outcome("abnormal", "疑似棉被遮擋口鼻，請立即查看", obstruction=covering)
        parts = [d for d in detections if self._inside(d, face)]
        noses = self._filter_by_labels(parts, self.nose_labels, self.part_confidence_threshold)
        mouths = self._filter_by_labels(parts, self.mouth_labels, self.part_confidence_threshold)
        pacifiers = self._filter_by_labels(parts, {"pacifier_in_mouth"}, self.part_confidence_threshold)
        blankets = self._filter_by_labels(detections, {"blanket"}, self.part_confidence_threshold)
        # A pacifier cannot exempt an unseen nose, or override any blanket evidence.
        missing = []
        if not noses:
            missing.append("nose")
        if not mouths and not pacifiers:
            missing.append("mouth")
        if missing:
            return outcome("unknown", "口鼻未能確認；未看見部位不等於已確認遮擋", missing_parts=missing)
        if any(self._overlaps(b, p) for b in blankets for p in noses + mouths + pacifiers):
            return outcome("unknown", "棉被偵測框與口鼻區重疊，需人工確認", missing_parts=[])
        if pacifiers and any(self._overlaps(p, n) for p in pacifiers for n in noses):
            return outcome("unknown", "奶嘴與鼻部區域重疊，無法確認鼻部是否暢通")
        return outcome("clear", "鼻部可見，奶嘴不列為棉被遮擋" if pacifiers else "本張鼻與嘴可見，未偵測到棉被遮擋證據",
                       pacifier_detected=bool(pacifiers), missing_parts=[])

    @staticmethod
    def _filter_by_labels(
        detections: list[dict[str, Any]],
        labels: set[str],
        threshold: float,
    ) -> list[dict[str, Any]]:
        # YOLO class 名稱轉小寫比對，讓 mouth/Mouth/MOUTH 都能被接受。
        normalized_labels = {label.lower() for label in labels}
        return [
            detection
            for detection in detections
            if str(detection.get("label", "")).lower() in normalized_labels
            and float(detection.get("confidence", 0)) >= threshold
        ]

    def _build_extra(
        self,
        detections: list[dict[str, Any]],
        visible_faces: list[dict[str, Any]],
        visible_noses: list[dict[str, Any]],
        visible_mouths: list[dict[str, Any]],
        frame_shape: tuple[int, int] | None,
    ) -> dict[str, Any]:
        # extra 保留偵測數量與原始 YOLO 結果，Detector Lab 會用來顯示判斷過程。
        extra = {
            "enabled": True,
            "model_path": self.model_path,
            "face_count": len(visible_faces),
            "nose_count": len(visible_noses),
            "mouth_count": len(visible_mouths),
            "face_confidence_threshold": self.face_confidence_threshold,
            "part_confidence_threshold": self.part_confidence_threshold,
            "detections": detections,
        }
        if frame_shape is not None:
            # 紀錄 detector 實際收到的影像尺寸，前端畫框時可用來做座標縮放。
            frame_height, frame_width = frame_shape
            extra["frame_width"] = int(frame_width)
            extra["frame_height"] = int(frame_height)
        return extra
