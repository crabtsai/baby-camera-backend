"""Experimental appearance/lost-localization warning, NOT an obstruction classifier.

Never learn a new reference while an existing one differs; that could learn away
an occlusion. A reference is a local visual sample, not a verified safe state.
"""
import time

import cv2
import numpy as np

from backend.schemas import DetectionResult


class FaceReferenceMonitor:
    def __init__(self):
        self.reference = None
        self.box = None
        self.stable = 0
        self.last_at = None
        self.part_regions = []
        self.candidate_box = None

    def observe(self, frame, parts, quality_ok, max_gap):
        now = time.monotonic()
        gap = self.last_at is not None and now-self.last_at > max_gap
        self.last_at = now
        if gap:
            self.stable = 0
        detections = parts.extra.get("detections", []) if parts and quality_ok else []
        box = next((d["box"] for d in detections if d["label"] == "face_estimate"), None)
        patch = None
        if box:
            h,w = frame.shape[:2]
            x1,y1,x2,y2 = [int(v) for v in box]
            if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h and min(x2-x1,y2-y1) >= 24:
                patch = cv2.resize(cv2.cvtColor(frame[y1:y2,x1:x2], cv2.COLOR_BGR2GRAY), (64,64)).astype(float)
        capture = False
        abnormal = False
        changed_ratio = None
        message = "等待連續三張臉部定位以建立本機參考；遮擋狀態未知"
        if self.reference is None:
            if patch is not None and self.candidate_box is not None:
                a, b = box, self.candidate_box
                overlap = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
                union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap
                if overlap/max(union,1) < .5:
                    self.stable = 0
            self.stable = self.stable+1 if patch is not None else 0
            self.candidate_box = box if patch is not None else None
            if self.stable >= 3:
                self.reference, self.box = patch, box
                for d in detections:
                    if d["label"] in {"nose_estimate", "mouth_estimate"}:
                        x1,y1,x2,y2 = d["box"]
                        self.part_regions.append((
                            max(0,int((x1-box[0])/(box[2]-box[0])*64)-3),
                            max(0,int((y1-box[1])/(box[3]-box[1])*64)-3),
                            min(64,int((x2-box[0])/(box[2]-box[0])*64)+3),
                            min(64,int((y2-box[1])/(box[3]-box[1])*64)+3)))
                capture = True
                message = "已建立臉部參考照片；不代表已確認口鼻暢通"
        elif patch is None:
            abnormal = True
            message = "臉部追蹤失效或畫面不清，無法排除遮擋，請立即查看"
        else:
            # Mean normalization tolerates moderate global brightness variation.
            delta = np.abs((patch-patch.mean())-(self.reference-self.reference.mean()))
            changed_ratio = float(np.mean(delta > 35))
            for x1,y1,x2,y2 in self.part_regions:
                if x2 > x1 and y2 > y1:
                    changed_ratio = max(changed_ratio, float(np.mean(delta[y1:y2,x1:x2] > 35)))
            abnormal = changed_ratio >= .35
            message = ("臉部外觀與參考明顯不同，可能是遮擋、轉頭或光線變化，請查看" if abnormal else
                       "臉部參考比對未達提醒門檻；未驗證口鼻是否被遮擋")
        return DetectionResult("face_reference_monitor", "face_reference_warning", abnormal, 0., message, {
            "evaluable": self.reference is not None,
            "assessment": "abnormal" if abnormal else "unknown",
            "reference_ready": self.reference is not None, "capture_reference": capture,
            "head": box, "reference_head": self.box, "changed_ratio": changed_ratio,
            "obstruction_confirmed": False, "experimental": True,
            "max_observation_gap_sec": max_gap, "observed_at": time.time(),
        })
