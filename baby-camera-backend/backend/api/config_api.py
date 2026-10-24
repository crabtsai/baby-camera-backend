from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/config")
def config(request: Request):
    # 對外回傳目前已載入的設定，方便確認 config.ini 修改後是否有重啟並生效。
    # 敏感資訊如 ingest_key 不回傳，避免前端或瀏覽器 console 洩漏。
    s = request.app.state.settings
    return {
        "app_name": s.app_name,
        "camera_id": s.camera_id,
        "camera_source": s.camera_source,
        "cameras": [
            {
                "id": camera.id,
                "source": camera.display_source,
                "source_type": camera.source_type,
            }
            for camera in s.cameras
        ],
        "frame_interval_sec": s.frame_interval_sec,
        "detectors": {
            "dark": s.enable_dark_detector,
            "motion": s.enable_motion_detector,
            "baby": s.enable_baby_detector,
            "face_cover": s.enable_face_cover_detector,
            "safety": s.enable_safety_detector,
        },
        "baby_detector": {
            "enabled": s.enable_baby_detector,
            "model_path": s.baby_yolo_model_path,
            "confidence_threshold": s.baby_confidence_threshold,
            "face_labels": s.baby_face_labels,
        },
        "face_cover_detector": {
            "enabled": s.enable_face_cover_detector,
            "model_path": s.face_cover_yolo_model_path,
            "face_confidence_threshold": s.face_cover_face_confidence_threshold,
            "part_confidence_threshold": s.face_cover_part_confidence_threshold,
            "require_nose": s.face_cover_require_nose,
            "require_mouth": s.face_cover_require_mouth,
        },
        "database_path": str(s.database_path),
        "snapshot_dir": str(s.snapshot_dir),
        "alert_webhook_enabled": bool(s.alert_webhook_url),
        "alert_dry_run": s.alert_dry_run,
    }
