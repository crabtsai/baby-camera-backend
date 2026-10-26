"""Offline face landmark localization; landmarks are not visibility evidence."""
from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from backend.detector.base_detector import BaseDetector
from backend.schemas import DetectionResult


class BabyFacePartsDetector(BaseDetector):
    name = "baby_face_parts_detector"
    model_id = "mediapipe-face-landmarker"

    def __init__(self, model_path="models/face_landmarker.task", confidence=.5, predict_fn=None):
        self.model_path = str(model_path)
        self.confidence = confidence
        self.last_results = []
        self.disabled_reason = None
        self.operational = False
        self._landmarker = None
        self._lock = threading.RLock()
        if not math.isfinite(confidence) or not 0 < confidence <= 1:
            self.disabled_reason = "口鼻模型信心門檻無效"
        elif predict_fn is None and not Path(self.model_path).is_file():
            self.disabled_reason = "缺少本機 MediaPipe 模型，請先下載 face_landmarker.task"
        self.predict_fn = None if self.disabled_reason else (predict_fn or self._predict)

    @classmethod
    def from_settings(cls, settings, camera_id=None):
        if settings is None:
            return cls()
        camera_id = camera_id or settings.camera_id
        return cls(
            model_path=settings.camera_detector_text(camera_id, "face_parts", "model_path", settings.face_parts_model_path),
            confidence=settings.camera_detector_float(camera_id, "face_parts", "confidence", settings.face_parts_confidence),
        )

    def _predict(self, frame):
        import mediapipe as mp
        if self._landmarker is None:
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(Path(self.model_path).resolve()),
                                                  delegate=mp.tasks.BaseOptions.Delegate.CPU),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=2,
                min_face_detection_confidence=self.confidence,
                min_face_presence_confidence=self.confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        result = self._landmarker.detect(image)
        return [[(p.x, p.y, p.z) for p in face] for face in result.face_landmarks]

    @staticmethod
    def localize(faces, shape):
        if len(faces) != 1:
            return [], [], "未找到單一臉部；無法定位口鼻"
        points = np.asarray(faces[0], dtype=float)
        if points.shape != (478, 3) or not np.isfinite(points).all():
            raise ValueError("Invalid landmarks")
        height, width = shape[:2]
        # Canonical mesh indices summarize estimated regions, not visible parts.
        regions = {"face_estimate": tuple(range(478)),
                   "nose_estimate": (1, 2, 98, 327, 168),
                   "mouth_estimate": (61, 291, 0, 17, 13, 14, 78, 308)}
        detections, keypoints = [], []
        for label, indices in regions.items():
            xy = points[list(indices), :2]
            if np.any(xy < 0) or np.any(xy > 1):
                return [], [], "臉部超出畫面，口鼻位置無法確認"
            pixel = xy * [width, height]
            low, high = pixel.min(axis=0), pixel.max(axis=0)
            if np.any(high <= low):
                return [], [], "關鍵點幾何無效，口鼻位置無法確認"
            detections.append({"label": label, "box": [*low.tolist(), *high.tolist()],
                               "confidence": 0., "estimated": True})
            if label != "face_estimate":
                keypoints.extend({"index": index, "x": float(points[index, 0] * width),
                                  "y": float(points[index, 1] * height)} for index in indices)
        return detections, keypoints, "已估計鼻部與嘴唇位置；不代表口鼻露出或未被遮擋"

    def detect(self, frame):
        with self._lock:
            detections, keypoints = [], []
            message = self.disabled_reason or "MediaPipe 推論失敗，口鼻位置無法確認"
            self.operational = False
            if self.predict_fn is not None:
                try:
                    faces = self.predict_fn(frame)
                    detections, keypoints, message = self.localize(faces, frame.shape)
                    self.operational = True
                except Exception:
                    detections, keypoints = [], []
            result = DetectionResult(self.name, "face_parts_visibility", False, 0., message, {
                "assessment": "unknown", "evaluable": False,
                "localization_status": "located" if detections else "unknown",
                "enabled": self.predict_fn is not None, "model_id": self.model_id,
                "offline": True, "obstruction_detection_supported": False,
                "visibility_confirmed": False, "detections": detections, "keypoints": keypoints,
                "frame_width": int(frame.shape[1]), "frame_height": int(frame.shape[0]),
                "observed_at": time.time(), "confidence_kind": "landmark_estimate_not_visibility",
            })
            self.last_results = [{"event_type": result.event_type, "message": message, **result.extra}]
            return result

    def close(self):
        with self._lock:
            if self._landmarker is not None:
                self._landmarker.close()
                self._landmarker = None
            self.operational = False

    def __del__(self):
        if hasattr(self, "_lock"):
            self.close()
