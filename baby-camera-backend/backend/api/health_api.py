from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request):
    app_state = request.app.state
    return {
        "status": "ok",
        "camera_id": app_state.settings.camera_id,
        "camera": app_state.camera.status(),
        "detectors": app_state.detector_manager.names(),
    }
