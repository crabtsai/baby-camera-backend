"""Top-down monitoring. One scene inference feeds all spatial rules.

Coordinates are normalized xyxy. Unknown is explicit and never evidence of safety.
Models and calibration must be validated against the actual camera before alerts.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from backend.detector.base_detector import BaseDetector
from backend.detector.occupancy import OccupancyTracker
from backend.schemas import DetectionResult

logger = logging.getLogger(__name__)
Box = tuple[float, float, float, float]


def parse_region(value: str) -> Box | None:
    if not value.strip():
        return None
    values = tuple(float(v.strip()) for v in value.split(","))
    if len(values) != 4 or not all(np.isfinite(v) for v in values):
        raise ValueError("sleep_region must contain four finite normalized coordinates")
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("sleep_region must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    return values


def intersection(a: Box, b: Box) -> float:
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))


def area(box: Box) -> float:
    return max(0., box[2] - box[0]) * max(0., box[3] - box[1])


def center_inside(box: Box, region: Box) -> bool:
    x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] <= x <= region[2] and region[1] <= y <= region[3]


@dataclass(frozen=True)
class SafetyOptions:
    region: Box | None = None
    boundary_margin: float = .08
    object_overlap: float = .1
    head_overlap: float = .15
    confidence: float = .5
    blur_threshold: float = 25.
    dark_threshold: float = 20.
    overexposed_ratio: float = .8
    hazardous_labels: tuple[str, ...] = ("pillow", "blanket", "toy", "teddy bear")
    present_frames: int = 3
    absent_frames: int = 5
    max_observation_gap_sec: float = 5.


class SceneModels:
    """Load only local weights; never download models during application startup."""

    def __init__(self, pose_path: str, object_path: str, posture_path: str, options: SafetyOptions):
        self.options = options
        self.capabilities: dict[str, dict] = {}
        self.pose = self._load("localization", pose_path, "pose")
        self.objects = self._load("objects", object_path, "detect", options.hazardous_labels)
        self.posture = self._load("posture", posture_path, "classify", ("supine", "side", "prone"))

    def _load(self, name, path, task, labels=()):
        reason = "未設定模型"
        model = None
        if path:
            try:
                if not Path(path).is_file():
                    raise ValueError("本機模型檔案不存在")
                from ultralytics import YOLO
                model = YOLO(path)
                if model.task != task:
                    raise ValueError(f"模型 task 必須是 {task}")
                names = {str(v).lower() for v in model.names.values()}
                missing = set(labels) - names
                if missing:
                    raise ValueError("模型缺少類別：" + ", ".join(sorted(missing)))
                reason = ""
            except Exception as exc:
                reason = str(exc)
                model = None
                logger.warning("Safety %s unavailable: %s", name, reason)
        self.capabilities[name] = {"available": model is not None, "reason": reason}
        return model

    def __call__(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        scene = {"people": [], "objects": [], "pose_available": self.pose is not None,
                 "objects_available": self.objects is not None}
        # Fail each model independently, so e.g. object inference still works if pose fails.
        if self.pose is not None:
            try:
                for result in self.pose.predict(frame, verbose=False, conf=self.options.confidence):
                    if result.boxes is not None and len(result.boxes) == 0:
                        continue
                    if result.boxes is None or result.keypoints is None or result.keypoints.conf is None:
                        raise ValueError("Pose 結果缺少可判讀的人體資料")
                    for box, xy, confidence in zip(result.boxes.xyxy.cpu().tolist(),
                                                   result.keypoints.xy.cpu().tolist(),
                                                   result.keypoints.conf.cpu().tolist()):
                        body = (box[0]/w, box[1]/h, box[2]/w, box[3]/h)
                        visible = [i for i in range(min(5, len(xy))) if confidence[i] >= self.options.confidence]
                        head = None
                        if len(visible) >= 2:
                            xs, ys = [xy[i][0]/w for i in visible], [xy[i][1]/h for i in visible]
                            # Approximate head region, not a segmentation mask or contact measurement.
                            pad = max(max(xs)-min(xs), max(ys)-min(ys), .02) * .5
                            head = (max(0., min(xs)-pad), max(0., min(ys)-pad),
                                    min(1., max(xs)+pad), min(1., max(ys)+pad))
                        scene["people"].append({"body": body, "head": head, "posture": None})
            except Exception as exc:
                scene.update(pose_available=False, pose_error=str(exc), people=[])
        if self.objects is not None:
            try:
                for result in self.objects.predict(frame, verbose=False, conf=self.options.confidence):
                    for box in (result.boxes if result.boxes is not None else []):
                        xy = box.xyxy[0].cpu().tolist()
                        scene["objects"].append({"label": str(result.names[int(box.cls[0].item())]).lower(),
                            "box": (xy[0]/w, xy[1]/h, xy[2]/w, xy[3]/h)})
            except Exception as exc:
                scene.update(objects_available=False, object_error=str(exc), objects=[])
        # Classify only the unambiguous person within the calibrated mattress.
        candidates = [p for p in scene["people"] if self.options.region and center_inside(p["body"], self.options.region)]
        if len(candidates) == 1 and self.posture is not None:
            try:
                p = candidates[0]
                x1, y1, x2, y2 = p["body"]
                crop = frame[max(0, int(y1*h)):min(h, int(y2*h)), max(0, int(x1*w)):min(w, int(x2*w))]
                result = self.posture.predict(crop, verbose=False)[0]
                if result.probs is not None and float(result.probs.top1conf) >= self.options.confidence:
                    p["posture"] = str(result.names[result.probs.top1]).lower()
            except Exception as exc:
                scene["posture_error"] = str(exc)
        return scene


class SafetyPipeline(BaseDetector):
    name = "safety_pipeline"
    EVENTS = ("sleep_area_object", "head_obstruction_warning", "boundary_warning", "sleep_posture_warning")
    DEPENDENT_EVENTS = {"motion_detector": "low_motion_warning", "baby_detector": "baby_position_warning",
                        "baby_face_parts_detector": "face_parts_visibility",
                        "face_cover_detector": "face_cover_warning", "prone_sleep_detector": "prone_sleep_warning"}

    def __init__(self, options: SafetyOptions | None = None, scene_fn: Callable | None = None):
        self.options = options or SafetyOptions()
        self.scene_fn = scene_fn
        self.capabilities = {"quality": {"available": True},
                             "calibration": {"available": self.options.region is not None,
                                             "reason": "" if self.options.region else "尚未設定床墊區域"},
                             **getattr(scene_fn, "capabilities", {})}
        self.last_results: list[dict] = []
        self._last_head: tuple[Box, Box, float] | None = None
        self.occupancy = OccupancyTracker(self.options.present_frames, self.options.absent_frames,
                                          self.options.max_observation_gap_sec)
        self.dependents: list[BaseDetector] = []

    @classmethod
    def from_settings(cls, settings, camera_id):
        def text(key, default=""):
            # Per-camera overrides fall back to shared safety settings.
            from backend.config import _get
            return settings.camera_detector_text(camera_id, "safety", key, _get("detectors.safety", key, default))

        def number(key, default, low, high):
            value = float(text(key, str(default)))
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"safety.{key} must be between {low} and {high}")
            return value

        opts = SafetyOptions(
            region=parse_region(text("sleep_region")),
            boundary_margin=number("boundary_margin", .08, 0, .49),
            object_overlap=number("object_overlap", .1, .001, 1),
            head_overlap=number("head_overlap", .15, .001, 1),
            confidence=number("confidence", .5, .01, 1),
            present_frames=int(number("present_frames", 3, 1, 100)),
            absent_frames=int(number("absent_frames", 5, 1, 100)),
            max_observation_gap_sec=max(5., settings.frame_interval_sec * 3),
            blur_threshold=number("blur_threshold", 25, 0, 10000),
            dark_threshold=number("dark_threshold", 20, 0, 255),
            overexposed_ratio=number("overexposed_ratio", .8, .01, 1),
            hazardous_labels=tuple(v.strip().lower() for v in text("hazardous_labels", "pillow,blanket,toy,teddy bear").split(",") if v.strip()),
        )
        if not opts.hazardous_labels:
            raise ValueError("safety.hazardous_labels cannot be empty")
        return cls(opts, SceneModels(text("pose_model_path"), text("object_model_path"), text("posture_model_path"), opts))

    def _result(self, event, abnormal=False, reason="", evaluable=True, **extra):
        return DetectionResult(self.name, event, abnormal, 0., reason,
            {"evaluable": evaluable, "assessment": "unknown" if not evaluable else "abnormal" if abnormal else "clear",
             "max_observation_gap_sec": self.options.max_observation_gap_sec, "confidence_kind": "rule_based_not_probability", **extra})

    def detect(self, frame):
        # Compatibility for callers expecting one result; use detect_many for all rules.
        return self.detect_many(frame)[0]

    def detect_many(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Fixed resolution keeps the blur threshold comparable across camera resolutions.
        small = cv2.resize(gray, (320, 240))
        brightness = float(gray.mean())
        sharpness = float(cv2.Laplacian(small, cv2.CV_64F).var())
        white_ratio = float(np.mean(gray >= 250))
        reasons = []
        if brightness < self.options.dark_threshold:
            reasons.append("畫面過暗")
        if sharpness < self.options.blur_threshold:
            reasons.append("畫面模糊或缺少可辨識細節")
        if white_ratio >= self.options.overexposed_ratio:
            reasons.append("畫面過曝")
        results = [self._result("image_quality_warning", bool(reasons), "、".join(reasons) or "畫面品質通過基本檢查",
                    brightness=brightness, sharpness=sharpness, overexposed_ratio=white_ratio)]
        scope = "sleep_region" if self.options.region else "frame"
        if reasons:
            self._last_head = None
            self.occupancy.observe(None, scope, "畫面品質不足，無法確認在床狀態")
            results.extend(self._result(e, reason="畫面品質不足，無法判讀", evaluable=False) for e in self.EVENTS)
        else:
            try:
                scene = self.scene_fn(frame) if self.scene_fn else {}
                candidates = [p for p in scene.get("people", []) if self.options.region is None
                              or center_inside(p["body"], self.options.region)]
                count = len(candidates) if scene.get("pose_available") else None
                self.occupancy.observe(count, scope, "定位模型不可用" if count is None else "")
                if self.options.region is None:
                    self._last_head = None
                    results.extend(self._result(e, reason="尚未設定床墊區域，無法判讀", evaluable=False) for e in self.EVENTS)
                else:
                    results.extend(self._assess(scene))
            except Exception:
                self._last_head = None
                self.occupancy.observe(None, scope, "定位推論失敗")
                logger.exception("Safety scene inference failed")
                results.extend(self._result(e, reason="場景推論失敗，無法判讀", evaluable=False) for e in self.EVENTS)
        occupancy = self.occupancy.status()
        results.insert(1, self._result("bed_occupancy", evaluable=occupancy["state"] != "unknown", **occupancy))
        # Run dependent rules behind THIS frame's presence gate, not a racing worker's last result.
        for detector in self.dependents:
            if occupancy["state"] == "present":
                try:
                    dependent_results = detector.detect_many(frame)
                    self.capabilities["dependent_detectors"][detector.name]["last_error"] = None
                    self.capabilities["dependent_detectors"][detector.name]["available"] = (
                        getattr(detector, "operational", True)
                        and not (hasattr(detector, "predict_fn") and detector.predict_fn is None)
                    )
                    for result in dependent_results:
                        result.extra.update(occupancy_state="present", max_observation_gap_sec=self.options.max_observation_gap_sec)
                        if result.extra.get("enabled") is False:
                            result.extra.update(evaluable=False, assessment="unknown")
                    results.extend(dependent_results)
                    continue
                except Exception as exc:
                    self.capabilities["dependent_detectors"][detector.name]["last_error"] = str(exc)
                    logger.exception("Presence-dependent detector failed: %s", detector.name)
            detector.reset()
            results.append(DetectionResult(detector.name, self.DEPENDENT_EVENTS[detector.name], False, 0.,
                "監測對象未確認或檢測不可用，暫停此項判讀",
                {"evaluable": False, "assessment": "unknown", "occupancy_state": occupancy["state"],
                 "max_observation_gap_sec": self.options.max_observation_gap_sec}))
        self.last_results = [{"event_type": r.event_type, "message": r.message, **r.extra} for r in results]
        return results

    def attach_dependents(self, detectors: list[BaseDetector]) -> list[BaseDetector]:
        """Move subject-specific detectors into the ordered pipeline; return independent ones."""
        self.dependents = [d for d in detectors if d.name in self.DEPENDENT_EVENTS]
        self.capabilities["dependent_detectors"] = {
            d.name: {"available": not (hasattr(d, "predict_fn") and d.predict_fn is None)} for d in self.dependents
        }
        return [d for d in detectors if d.name not in self.DEPENDENT_EVENTS]

    @property
    def operational(self) -> bool:
        localization = self.capabilities.get("localization", {"available": self.scene_fn is not None})
        return localization["available"] and all(
            status["available"] and not status.get("last_error")
            for status in self.capabilities.get("dependent_detectors", {}).values()
        )

    def _assess(self, scene):
        region = self.options.region
        people = [p for p in scene.get("people", []) if center_inside(p["body"], region)]
        person = people[0] if scene.get("pose_available") and len(people) == 1 and self.occupancy.status()["state"] == "present" else None
        objects_ready = scene.get("objects_available", False)
        hazards = [o for o in scene.get("objects", []) if o["label"] in self.options.hazardous_labels
                   and intersection(o["box"], region) / max(area(o["box"]), 1e-9) >= self.options.object_overlap]
        yield self._result("sleep_area_object", bool(hazards) if objects_ready else False,
            "睡眠區出現需留意物件" if hazards and objects_ready else "未發現設定類別的物件" if objects_ready else "異物模型不可用",
            evaluable=objects_ready, objects=hazards, sleep_region=region)

        head = person.get("head") if person else None
        head_source = "current" if head else "unknown"
        now = time.monotonic()
        if person and head:
            self._last_head = (person["body"], head, now)
        elif person and self._last_head:
            old_body, old_head, seen_at = self._last_head
            overlap = intersection(old_body, person["body"])
            iou = overlap / max(area(old_body) + area(person["body"]) - overlap, 1e-9)
            if now - seen_at <= 3 and iou >= .7:
                head, head_source = old_head, "recent_estimate"
            else:
                self._last_head = None
        else:
            self._last_head = None
        head_ready = objects_ready and head is not None
        overlapping = [o for o in hazards if head_ready and intersection(o["box"], head) / max(area(head), 1e-9) >= self.options.head_overlap]
        # A remembered position can support a warning, but cannot prove the current head is clear.
        if head_source == "recent_estimate" and not overlapping:
            head_ready = False
        yield self._result("head_obstruction_warning", bool(overlapping),
            "物件與頭部區域重疊，請查看畫面" if overlapping else "未發現物件與頭部重疊" if head_ready else "無法確認頭部或異物位置",
            evaluable=head_ready, head=head, head_source=head_source, objects=overlapping)

        if person:
            body = person["body"]
            mx = (region[2]-region[0]) * self.options.boundary_margin
            my = (region[3]-region[1]) * self.options.boundary_margin
            near = body[0] <= region[0]+mx or body[1] <= region[1]+my or body[2] >= region[2]-mx or body[3] >= region[3]-my
            yield self._result("boundary_warning", near, "身體接近設定邊界" if near else "身體位於設定範圍內",
                               body=body, sleep_region=region)
        else:
            yield self._result("boundary_warning", reason="床墊內無可辨識人物或有多人，無法判讀", evaluable=False)

        posture = person.get("posture") if person else None
        ready = posture in {"supine", "side", "prone"}
        yield self._result("sleep_posture_warning", posture in {"side", "prone"},
            f"目前姿勢：{posture}" if ready else "睡姿模型不可用或無法確認姿勢", evaluable=ready, posture=posture)
