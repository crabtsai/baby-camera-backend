"""Bounded, local, unlabelled learning queue; never changes detector weights."""
import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2

from backend.services.snapshot_service import SnapshotService


class BlanketCollection:
    def __init__(self, camera_id, enabled=False, interval=60, limit=1000,
                 root=Path("data/airway/blanket/pending")):
        self.directory=Path(root)/SnapshotService._safe_name(camera_id)
        self.camera_id=camera_id
        self.enabled=enabled
        self.interval=max(1.,float(interval))
        self.limit=max(1,int(limit))
        self.last_at=None
        self.present=False
        self.last_path=None
        self.error=None
        self.count=len(list(self.directory.glob("*.jpg"))) if enabled else 0
        self.session=uuid4().hex
        self.pending_frame_hash=None

    def observe(self, frame, result):
        if not self.enabled:return
        if result.event_type=="sleep_area_object" and self.pending_frame_hash and "objects" in result.extra:
            # Consume only the same image used for collection, never a later frame's masks.
            digest=hashlib.sha256(frame.tobytes()).hexdigest()
            if digest==self.pending_frame_hash and self.last_path:
                try:
                    objects=[o for o in result.extra["objects"] if o.get("label")=="blanket" and "polygon" in o]
                    Path(self.last_path).with_suffix(".proposal.json").write_text(json.dumps({
                        "reviewed":False,"source":"runtime_candidate_segmentation",
                        "polygons":[o["polygon"] for o in objects],
                        "confidence":[o.get("confidence",0) for o in objects],
                        "note":"Machine proposals only; require review before training"}),encoding="utf-8")
                except OSError:
                    self.error="棉被建議輪廓儲存失敗，照片仍保留"
            self.pending_frame_hash=None
            return
        if result.event_type!="bed_occupancy":return
        state=result.extra.get("state")
        if state=="absent":
            if self.present:self.session=uuid4().hex
            self.present=False
            return
        if state!="present":return
        now=time.monotonic()
        new_presence=not self.present
        self.present=True
        if self.count>=self.limit:
            self.error="待標註照片已達上限，請整理資料後重啟收集"
            return
        if not new_presence and self.last_at is not None and now-self.last_at<self.interval:return
        # Limit failed disk writes too, while leaving the monitoring pipeline running.
        self.last_at=now
        try:
            self.directory.mkdir(parents=True,exist_ok=True)
            path=self.directory/(datetime.now().strftime("%Y%m%d_%H%M%S_%f")+"_"+uuid4().hex[:8]+".jpg")
            if not cv2.imwrite(str(path),frame):raise OSError("Image write failed")
            self.count+=1
            path.with_suffix(".json").write_text(json.dumps({
                "camera_id":self.camera_id,"session":self.session,"captured_at":datetime.now().isoformat(),
                "reviewed":False,"identity_verified":False,"reason":"confirmed_person_candidate",
                "purpose":"blanket_segmentation_annotation"},ensure_ascii=False),encoding="utf-8")
            self.last_path=str(path)
            self.pending_frame_hash=hashlib.sha256(frame.tobytes()).hexdigest()
            self.error=None
        except (OSError,cv2.error):
            self.error="學習照片儲存失敗；即時監測仍繼續"

    def status(self):
        return {"enabled":self.enabled,"count":self.count,"limit":self.limit,
                "interval_sec":self.interval,"latest":self.last_path,"error":self.error,
                "training_status":"awaiting_reviewed_labels"}
