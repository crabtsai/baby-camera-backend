"""Visibility assessment with sho's baby-face-parts/2 Roboflow model.

This model localizes anatomy, not blankets or airway obstruction.
"""
from __future__ import annotations

import base64
import math
import time
from urllib.parse import urlsplit

import cv2
import requests

from backend.detector.base_detector import BaseDetector
from backend.detector.face_cover import FaceCoverDetector
from backend.schemas import DetectionResult


class BabyFacePartsDetector(BaseDetector):
    name = "baby_face_parts_detector"
    model_id = "baby-face-parts/2"

    def __init__(self, api_key="", api_url="http://127.0.0.1:9001", confidence=.4,
                 timeout_sec=10., allow_cloud=False, predict_fn=None):
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.confidence = confidence
        self.timeout_sec = timeout_sec
        self.last_results = []
        self.disabled_reason = None
        self.operational = False
        url = urlsplit(self.api_url)
        local = (url.scheme == "http" and url.hostname in {"127.0.0.1", "::1", "localhost"})
        cloud = allow_cloud and self.api_url == "https://serverless.roboflow.com"
        if url.username or url.password or url.query or url.fragment or url.path not in {"", "/"} or not (local or cloud):
            self.disabled_reason = "口鼻模型僅允許本機服務；Roboflow 雲端需另行同意並啟用"
        elif not math.isfinite(confidence) or not 0 < confidence <= 1 or not math.isfinite(timeout_sec) or not 0 < timeout_sec <= 60:
            self.disabled_reason = "口鼻模型信心門檻或逾時設定無效"
        elif not api_key and predict_fn is None:
            self.disabled_reason = "尚未設定 Roboflow API key，口鼻模型未連接"
        self.session = requests.Session()
        self.session.trust_env = False  # Do not send frames through environment proxies.
        self.predict_fn = None if self.disabled_reason else (predict_fn or self._predict)

    @classmethod
    def from_settings(cls, settings, camera_id=None):
        if settings is None:
            return cls()
        camera_id = camera_id or settings.camera_id
        return cls(
            api_key=settings.face_parts_api_key,
            api_url=settings.face_parts_api_url,
            allow_cloud=settings.face_parts_allow_cloud,
            confidence=settings.camera_detector_float(camera_id, "face_parts", "confidence", settings.face_parts_confidence),
            timeout_sec=settings.face_parts_timeout_sec,
        )

    def _predict(self, frame):
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError("JPEG encoding failed")
        response = self.session.post(
            self.api_url + "/infer/object_detection",
            headers={"Authorization": "Bearer " + self.api_key},
            json={"model_id": self.model_id, "confidence": self.confidence,
                  "image": {"type": "base64", "value": base64.b64encode(encoded).decode("ascii")},
                  "disable_model_monitoring": True},
            timeout=(3., self.timeout_sec), allow_redirects=False,
        )
        if response.status_code != 200:
            # Never surface response text or request URLs containing secrets.
            raise ValueError("Inference service unavailable")
        return self.normalize(response.json())

    @staticmethod
    def normalize(payload):
        if not isinstance(payload, dict) or not isinstance(payload.get("predictions"), list):
            raise ValueError("Invalid prediction payload")
        detections = []
        for prediction in payload["predictions"]:
            label = str(prediction["class"]).upper()
            if label not in {"HEAD", "EYE", "NOSE", "MOUTH", "BABY"}:
                raise ValueError("Unexpected model classes")
            x, y, w, h, confidence = (float(prediction[key]) for key in ("x", "y", "width", "height", "confidence"))
            if not all(math.isfinite(v) for v in (x, y, w, h, confidence)) or w <= 0 or h <= 0 or not 0 <= confidence <= 1:
                raise ValueError("Invalid prediction box")
            detections.append({"label": label.lower(), "confidence": confidence,
                               "box": [x-w/2, y-h/2, x+w/2, y+h/2]})
        return detections

    def detect(self, frame):
        detections, state, missing = [], "unknown", []
        message = self.disabled_reason or "口鼻模型服務不可用，請確認本機服務與 API key"
        self.operational = False
        if self.predict_fn is not None:
            try:
                detections = self.predict_fn(frame)
                detections = [d for d in detections if d["confidence"] >= self.confidence]
                heads = [d for d in detections if d["label"] == "head" and FaceCoverDetector._box(d)]
                babies = [d for d in detections if d["label"] == "baby"]
                self.operational = True
                if len(heads) != 1 or len(babies) > 1:
                    message = "頭部未確認或有多個監測對象，無法判讀口鼻"
                else:
                    parts = {d["label"] for d in detections if FaceCoverDetector._inside(d, heads[0])}
                    missing = [part for part in ("nose", "mouth") if part not in parts]
                    if missing:
                        message = "未確認" + "、".join({"nose": "鼻部", "mouth": "嘴部"}[p] for p in missing) + "；不代表已確認被子遮擋"
                    else:
                        state = "clear"
                        message = "本張偵測到鼻部與嘴部；此模型不辨識被子或奶嘴"
            except Exception:
                # Return unknown without retaining a preceding successful result.
                detections = []
                self.operational = False
        result = DetectionResult(self.name, "face_parts_visibility", False, 0., message, {
            "assessment": state, "evaluable": state != "unknown", "enabled": self.predict_fn is not None,
            "model_id": self.model_id, "obstruction_detection_supported": False,
            "detections": detections, "missing_parts": missing,
            "frame_width": int(frame.shape[1]), "frame_height": int(frame.shape[0]),
            "observed_at": time.time(), "confidence_kind": "rule_based_not_probability",
        })
        self.last_results = [{"event_type": result.event_type, "message": message, **result.extra}]
        return result
