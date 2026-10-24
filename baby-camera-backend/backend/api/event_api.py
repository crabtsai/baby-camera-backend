from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Query, Request

router = APIRouter()


@router.get("/events")
def list_events(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    camera_id: str | None = Query(default=None),
):
    # 查詢近期事件紀錄，Detector Lab 或後續管理頁可用這個 API 顯示歷史告警。
    return request.app.state.database.list_events(limit=limit, camera_id=camera_id)


@router.post("/test-alert")
def test_alert(request: Request, camera_id: str | None = Query(default=None)):
    # 手動建立測試事件，用來確認 SQLite 紀錄與 Firebase/LINE 告警通道是否正常。
    runtime = request.app.state.camera_manager.get(camera_id)
    if runtime is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"找不到相機：{camera_id}")
    event = runtime.event_manager.create_test_event()
    return {
        "ok": True,
        "event": asdict(event),
    }
