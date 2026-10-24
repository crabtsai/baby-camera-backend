"""Per-camera presence evidence, without assuming infant identity or a fixed bed."""
from __future__ import annotations

import time


class OccupancyTracker:
    def __init__(self, present_frames=3, absent_frames=5, max_gap_sec=5.):
        self.present_frames = present_frames
        self.absent_frames = absent_frames
        self.max_gap_sec = max_gap_sec
        self._candidate = None
        self._count = 0
        self._last_at = None
        self._status = {"state": "unknown", "reason": "等待影像", "scope": "frame",
                        "identity_verified": False, "consecutive_frames": 0}

    def observe(self, count: int | None, scope: str, reason: str = "") -> dict:
        now = time.monotonic()
        if self._last_at is None or now - self._last_at > self.max_gap_sec:
            self._candidate, self._count = None, 0
        self._last_at = now
        candidate = "present" if count == 1 else "absent" if count == 0 else None
        if candidate is None:
            self._candidate, self._count = None, 0
        elif candidate == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = candidate, 1
        threshold = self.present_frames if candidate == "present" else self.absent_frames
        state = candidate if candidate and self._count >= threshold else "unknown"
        if not reason:
            reason = ("持續偵測到單一人物候選，尚未確認寶寶身分" if state == "present" else
                      "連續未偵測到人物；不代表已確認空床" if state == "absent" else
                      "多個人物，無法確定監測對象" if count is not None and count > 1 else
                      "等待連續影像確認")
        self._status = {"state": state, "reason": reason, "scope": scope,
                        "identity_verified": False, "person_count": count,
                        "consecutive_frames": self._count, "updated_at": time.time()}
        return dict(self._status)

    def status(self) -> dict:
        status = dict(self._status)
        if self._last_at is None or time.monotonic() - self._last_at > self.max_gap_sec:
            status.update(state="unknown", reason="尚無有效觀測或結果已過期")
        return status


def summarize_occupancy(cameras: list[dict]) -> dict:
    present, absent, unknown = [], [], []
    for camera in cameras:
        camera_id = camera.get("camera_id")
        if not camera_id:
            continue
        state = camera.get("occupancy", {}).get("state", "unknown")
        if not camera.get("camera", {}).get("frame_fresh", False):
            state = "unknown"
        (present if state == "present" else absent if state == "absent" else unknown).append(camera_id)
    selection = ("multiple_candidates" if len(present) > 1 else "uncertain" if unknown else
                 "single_candidate" if len(present) == 1 else "not_detected")
    return {"selection": selection, "candidate_camera_ids": present, "not_detected_camera_ids": absent,
            "unknown_camera_ids": unknown, "identity_verified": False,
            "suggested_camera_id": present[0] if selection == "single_candidate" else None}
