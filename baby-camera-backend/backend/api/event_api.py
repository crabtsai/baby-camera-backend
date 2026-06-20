from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Query, Request

router = APIRouter()


@router.get("/events")
def list_events(request: Request, limit: int = Query(default=50, ge=1, le=500)):
    return request.app.state.database.list_events(limit=limit)


@router.post("/test-alert")
def test_alert(request: Request):
    event = request.app.state.event_manager.create_test_event()
    return {
        "ok": True,
        "event": asdict(event),
    }
