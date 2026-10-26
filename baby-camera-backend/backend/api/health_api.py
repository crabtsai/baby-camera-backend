from __future__ import annotations

from fastapi import APIRouter, Request
from backend.detector.occupancy import summarize_occupancy

router = APIRouter()


@router.get("/health")
def health(request: Request):
    # 健康檢查入口：給前端頁面或人工測試確認後端、攝影機、detector 狀態。
    app_state = request.app.state
    default_runtime = app_state.camera_manager.default_runtime()
    cameras = app_state.camera_manager.list_status()
    return {
        "status": "ok" if cameras and all(
            c["camera"].get("frame_fresh", False)
            and (not c.get("high_frequency_motion",{}).get("enabled")
                 or c["high_frequency_motion"].get("alive",False))
            and all(w["alive"] and w.get("operational", True) and not w["last_error"]
                    for w in c["detector_workers"])
            for c in cameras) else "degraded",
        "camera_id": default_runtime.camera_id,
        "default_camera_id": app_state.camera_manager.default_camera_id,
        # 舊版相容欄位：仍代表預設相機。
        "camera": default_runtime.camera.status(),
        "detectors": default_runtime.detector_manager.names(),
        "detector_workers": default_runtime.detector_manager.worker_status(),
        # 多相機欄位：前端可用這裡列出所有相機與 worker 狀態。
        "cameras": cameras,
        "occupancy_summary": summarize_occupancy(cameras),
        "audio": app_state.audio_monitor.status() if hasattr(app_state, "audio_monitor") else {"enabled": False},
    }
