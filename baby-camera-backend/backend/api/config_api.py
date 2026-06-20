from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/config")
def config(request: Request):
    s = request.app.state.settings
    return {
        "app_name": s.app_name,
        "camera_id": s.camera_id,
        "camera_source": s.camera_source,
        "frame_interval_sec": s.frame_interval_sec,
        "detectors": {
            "dark": s.enable_dark_detector,
            "motion": s.enable_motion_detector,
            "baby": s.enable_baby_detector,
            "face_cover": s.enable_face_cover_detector,
        },
        "database_path": str(s.database_path),
        "snapshot_dir": str(s.snapshot_dir),
        "alert_webhook_enabled": bool(s.alert_webhook_url),
        "alert_dry_run": s.alert_dry_run,
    }
